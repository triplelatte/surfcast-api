from __future__ import annotations

import os
import time
import hashlib
from typing import Any, Optional, Tuple

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
# Models
# -----------------------------
class ForecastResponse(BaseModel):
    spot: dict
    window_hours: int
    generated_at: str
    best_windows: list
    hourly: list
    notes: list[str]

# -----------------------------
# Config
# -----------------------------
SPOTS_CSV_PATH = os.getenv("SPOTS_CSV_PATH", "./surf_registry_us_v4_with_latlon.csv")

OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"
OPEN_METEO_WEATHER = "https://api.open-meteo.com/v1/forecast"

app = FastAPI(title="SurfCast API", version="0.2.0")

# -----------------------------
# Load Spot Registry
# -----------------------------
spots_df = pd.read_csv(SPOTS_CSV_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "spots_loaded": len(spots_df)}

@app.get("/v1/spots/search")
def search_spots(q: str = Query(..., min_length=2), limit: int = 20):
    matches = spots_df[spots_df["spot_name"].str.contains(q, case=False, na=False)].head(limit)
    return matches[["spot_id", "spot_name", "state", "region"]].to_dict(orient="records")

async def geocode(query: str, country_code: str = "US") -> Tuple[float, float]:
    key = stable_key("geo", query, country_code)
    cached = cache_get(key)
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            OPEN_METEO_GEO,
            params={"name": query, "count": 1, "format": "json", "countryCode": country_code},
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results") or []
    if not results:
        raise HTTPException(404, f"Could not geocode: {query}")

    lat = float(results[0]["latitude"])
    lon = float(results[0]["longitude"])

    cache_set(key, (lat, lon), ttl_seconds=30 * 24 * 3600)
    return lat, lon

def surf_estimate_ft(swell_m, period_s, wind_wave_m, wind_mps):
    if swell_m is None or period_s is None:
        return None

    energy = swell_m * (max(period_s, 1) ** 0.5)
    chop = wind_wave_m or 0
    face_ft = 3.5 * energy - 2.0 * chop

    if wind_mps and wind_mps > 8:
        face_ft *= 0.8

    return max(face_ft, 0)

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

    # Prefer stored lat/lon (best)
    lat = spot.get("lat")
    lon = spot.get("lon")

    if pd.notna(lat) and pd.notna(lon):
        lat, lon = float(lat), float(lon)
        notes_extra = "Using stored lat/lon from registry."
    else:
        # Fallback: geocode (should be rare once registry is enriched)
        lat, lon = await geocode(str(spot["geocode_query"]))
        notes_extra = "Lat/lon not in registry; geocoded at runtime."

    # Cache per-hour bucket to avoid pounding upstream
    cache_key = stable_key("forecast", spot_id, str(hours), timezone, str(int(time.time() // 3600)))
    cached = cache_get(cache_key)
    if cached:
        return cached

    hourly_vars = ",".join([
        "wave_height,wave_period,wave_direction",
        "swell_wave_height,swell_wave_period,swell_wave_direction",
        "wind_wave_height",
    ])

    async with httpx.AsyncClient(timeout=25) as client:
        marine = await client.get(
            OPEN_METEO_MARINE,
            params={"latitude": lat, "longitude": lon, "hourly": hourly_vars, "timezone": timezone},
        )
        marine.raise_for_status()
        marine_json = marine.json()

        weather = await client.get(
            OPEN_METEO_WEATHER,
            params={"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m", "timezone": timezone},
        )
        weather.raise_for_status()
        weather_json = weather.json()

    times = marine_json["hourly"]["time"][:hours]

    hourly_data = []
    for i, t in enumerate(times):
        swell = marine_json["hourly"]["swell_wave_height"][i]
        period = marine_json["hourly"]["swell_wave_period"][i]
        wind_wave = marine_json["hourly"]["wind_wave_height"][i]
        wind = weather_json["hourly"]["wind_speed_10m"][i]

        est = surf_estimate_ft(swell, period, wind_wave, wind)

        hourly_data.append({
            "time": t,
            "swell_m": swell,
            "period_s": period,
            "wind_mps": wind,
            "surf_est_ft": est
        })

    # Rank best 3-hour windows
    best_windows = []
    scores = []
    for i in range(len(hourly_data) - 2):
        block = hourly_data[i:i+3]
        vals = [h["surf_est_ft"] for h in block if h["surf_est_ft"] is not None]
        if len(vals) < 2:
            continue
        avg = sum(vals) / len(vals)
        scores.append((avg, i))

    scores.sort(reverse=True)
    for avg, i in scores[:3]:
        block = hourly_data[i:i+3]
        best_windows.append({
            "start": block[0]["time"],
            "end": block[-1]["time"],
            "surf_est_ft_range": [
                round(min(h["surf_est_ft"] for h in block), 1),
                round(max(h["surf_est_ft"] for h in block), 1),
            ],
            "summary": "Top 3-hour surf window"
        })

    response = {
        "spot": {
            "spot_id": spot["spot_id"],
            "name": spot["spot_name"],
            "state": spot["state"],
            "region": spot["region"],
            "lat": lat,
            "lon": lon
        },
        "window_hours": hours,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "best_windows": best_windows,
        "hourly": hourly_data,
        "notes": [
            notes_extra,
            "Surf estimate is spot-agnostic; reefs/points/bathymetry can amplify or shadow swell.",
            "Wind direction vs beach orientation not yet modeled."
        ]
    }

    cache_set(cache_key, response, ttl_seconds=1800)
    return response
