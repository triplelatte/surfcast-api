from __future__ import annotations

import os
import time
import hashlib
import asyncio
from typing import Any, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


# -----------------------------
# Simple in-memory TTL cache
# -----------------------------
_CACHE: dict[str, tuple[float, Any]] = {}

def cache_get(key: str) -> Optional[Any]:
    item = _CACHE.get(key)
    if not item:
        return None
    exp, val = item
    if time.time() > exp:
        _CACHE.pop(key, None)
        return None
    return val

def cache_set(key: str, val: Any, ttl_seconds: int) -> None:
    _CACHE[key] = (time.time() + ttl_seconds, val)

def stable_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


# -----------------------------
# Open-Meteo endpoints
# -----------------------------
OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"
OPEN_METEO_WEATHER = "https://api.open-meteo.com/v1/forecast"


# -----------------------------
# FastAPI app
# -----------------------------
class ForecastResponse(BaseModel):
    spot: dict
    window_hours: int
    generated_at: str
    best_windows: list
    hourly: list
    notes: list[str]

app = FastAPI(title="SurfCast API", version="0.3.0")


# -----------------------------
# Load Spot Registry
# -----------------------------
SPOTS_CSV_PATH = os.getenv("SPOTS_CSV_PATH", "./surf_registry_us_v4_with_latlon.csv")
spots_df = pd.read_csv(SPOTS_CSV_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "spots_loaded": int(len(spots_df))}


@app.get("/v1/spots/search")
def search_spots(q: str = Query(..., min_length=2), limit: int = 20):
    matches = spots_df[
        spots_df["spot_name"].str.contains(q, case=False, na=False)
    ].head(limit)

    return matches[["spot_id", "spot_name", "state", "region"]].to_dict(orient="records")


# -----------------------------
# HTTP helper with retry/backoff
# -----------------------------
async def get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    retries: int = 4,
    base_delay_s: float = 1.0,
):
    """
    Retries on 429 with exponential backoff.
    """
    delay = base_delay_s
    last_status = None

    for attempt in range(retries + 1):
        resp = await client.get(url, params=params)
        last_status = resp.status_code

        if resp.status_code != 429:
            resp.raise_for_status()
            return resp

        # 429: Too Many Requests
        if attempt == retries:
            # include a helpful message
            raise HTTPException(
                status_code=429,
                detail=f"Upstream rate limited (429) from Open-Meteo after {retries+1} attempts: {url}"
            )

        await asyncio.sleep(delay)
        delay *= 2

    raise HTTPException(status_code=502, detail=f"Unexpected upstream error (status={last_status}).")


# -----------------------------
# Surf estimate heuristic
# -----------------------------
def surf_estimate_ft(swell_m, period_s, wind_wave_m, wind_mps):
    if swell_m is None or period_s is None:
        return None

    energy = float(swell_m) * (max(float(period_s), 1.0) ** 0.5)
    chop = float(wind_wave_m or 0.0)
    face_ft = 3.5 * energy - 2.0 * chop

    # penalize very windy conditions even without direction
    if wind_mps is not None and float(wind_mps) > 8.0:  # ~18mph
        face_ft *= 0.8

    return max(face_ft, 0.0)


@app.get("/v1/forecast", response_model=ForecastResponse)
async def forecast(
    spot_id: str,
    hours: int = Query(72, ge=6, le=168),
    timezone: str = "America/Los_Angeles",
):
    row = spots_df[spots_df["spot_id"] == spot_id]
    if row.empty:
        raise HTTPException(404, "Spot not found")

    spot = row.iloc[0]

    # Prefer stored lat/lon
    lat = spot.get("lat")
    lon = spot.get("lon")
    if pd.isna(lat) or pd.isna(lon):
        raise HTTPException(
            status_code=400,
            detail="Spot is missing lat/lon in the registry. Please enrich the CSV with coordinates."
        )

    lat = float(lat)
    lon = float(lon)

    # Hour-bucket cache: one upstream fetch per spot per hour (per hours/timezone settings)
    hour_bucket = str(int(time.time() // 3600))
    cache_key = stable_key("forecast", spot_id, str(hours), timezone, hour_bucket)

    cached = cache_get(cache_key)
    if cached:
        return cached

    hourly_marine = ",".join([
        "wave_height,wave_period,wave_direction",
        "swell_wave_height,swell_wave_period,swell_wave_direction",
        "wind_wave_height",
    ])

    async with httpx.AsyncClient(timeout=30) as client:
        marine_resp = await get_with_backoff(
            client,
            OPEN_METEO_MARINE,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": hourly_marine,
                "timezone": timezone,
            },
        )
        marine_json = marine_resp.json()

        weather_resp = await get_with_backoff(
            client,
            OPEN_METEO_WEATHER,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "wind_speed_10m",
                "timezone": timezone,
            },
        )
        weather_json = weather_resp.json()

    times = (marine_json.get("hourly", {}).get("time") or [])[:hours]
    if not times:
        raise HTTPException(502, "Upstream returned no hourly time series.")

    # Pull arrays (safely)
    H_total = (marine_json["hourly"].get("wave_height") or [])[:hours]
    H_swell = (marine_json["hourly"].get("swell_wave_height") or [])[:hours]
    T_swell = (marine_json["hourly"].get("swell_wave_period") or [])[:hours]
    H_windwave = (marine_json["hourly"].get("wind_wave_height") or [])[:hours]
    wind = (weather_json.get("hourly", {}).get("wind_speed_10m") or [])[:hours]

    hourly_data = []
    n = min(len(times), len(H_total), len(H_swell), len(T_swell), len(H_windwave), len(wind))

    for i in range(n):
        est = surf_estimate_ft(H_swell[i], T_swell[i], H_windwave[i], wind[i])
        hourly_data.append({
            "time": times[i],
            "total_wave_m": H_total[i],
            "swell_m": H_swell[i],
            "period_s": T_swell[i],
            "wind_mps": wind[i],
            "surf_est_ft": est,
        })

    # Rank best 3-hour windows by estimated surf face height
    best_windows = []
    scores = []
    for i in range(len(hourly_data) - 2):
        block = hourly_data[i:i+3]
        vals = [h["surf_est_ft"] for h in block if h["surf_est_ft"] is not None]
        if len(vals) < 2:
            continue
        scores.append((sum(vals) / len(vals), i))

    scores.sort(reverse=True)
    for avg, i in scores[:3]:
        block = hourly_data[i:i+3]
        best_windows.append({
            "start": block[0]["time"],
            "end": block[-1]["time"],
            "surf_est_ft_range": [
                round(min(h["surf_est_ft"] for h in block if h["surf_est_ft"] is not None), 1),
                round(max(h["surf_est_ft"] for h in block if h["surf_est_ft"] is not None), 1),
            ],
            "summary": "Top 3-hour surf window (spot-agnostic estimate).",
        })

    response = {
        "spot": {
            "spot_id": spot["spot_id"],
            "name": spot["spot_name"],
            "state": spot["state"],
            "region": spot["region"],
            "lat": lat,
            "lon": lon,
        },
        "window_hours": hours,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "best_windows": best_windows,
        "hourly": hourly_data,
        "notes": [
            "Using stored lat/lon from registry; no runtime geocoding.",
            "Hourly data is cached per-spot per-hour to reduce upstream load.",
            "Surf estimate is spot-agnostic; local bathymetry/sandbars can change outcomes.",
        ],
    }

    # cache for 55 minutes so it survives most of the hour bucket
    cache_set(cache_key, response, ttl_seconds=55 * 60)
    return response