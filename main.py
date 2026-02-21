from __future__ import annotations

import os
import time
import hashlib
import asyncio
from typing import Any, Optional
from datetime import datetime

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


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


OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"


class ForecastResponse(BaseModel):
    spot: dict
    window_hours: int
    generated_at: str
    best_windows: list
    hourly: list
    notes: list[str]


app = FastAPI(title="SurfCast API", version="0.5.0")


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


async def get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    retries: int = 4,
    base_delay_s: float = 1.0,
):
    delay: float = float(base_delay_s)

    for attempt in range(retries + 1):
        resp = await client.get(url, params=params)

        if resp.status_code != 429:
            resp.raise_for_status()
            return resp

        if attempt == retries:
            raise HTTPException(
                status_code=429,
                detail="Upstream rate limited (429) from Open-Meteo marine endpoint."
            )

        await asyncio.sleep(delay)
        delay = delay * 2.0


def surf_estimate_ft(swell_m, period_s, wind_wave_m):
    if swell_m is None or period_s is None:
        return None

    energy = float(swell_m) * (max(float(period_s), 1.0) ** 0.5)
    chop = float(wind_wave_m or 0.0)
    face_ft = 3.5 * energy - 2.0 * chop
    return max(face_ft, 0.0)


def find_start_index(times: list[str], timezone: str) -> int:
    """
    Given Open-Meteo hourly time strings like 'YYYY-MM-DDTHH:MM' in the requested timezone,
    find the first index whose time is >= current local hour.
    """
    if ZoneInfo is None:
        # Fallback: just start at 0 if zoneinfo isn't available
        return 0

    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    now_floor = now.replace(minute=0, second=0, microsecond=0)

    # Fast path: match exact current-hour string
    now_key = now_floor.strftime("%Y-%m-%dT%H:%M")
    try:
        return times.index(now_key)
    except ValueError:
        pass

    # Otherwise find first time >= now_floor
    for i, t in enumerate(times):
        # times are local time strings without offset; interpret in tz
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=tz)
        except ValueError:
            # If any weird formatting slips in, skip it
            continue
        if dt >= now_floor:
            return i

    return 0


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

    lat = float(spot["lat"])
    lon = float(spot["lon"])

    # Cache per hour bucket (still fine)
    hour_bucket = str(int(time.time() // 3600))
    cache_key = stable_key("forecast", spot_id, str(hours), timezone, hour_bucket)

    cached = cache_get(cache_key)
    if cached:
        return cached

    hourly_vars = ",".join([
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
                "hourly": hourly_vars,
                "timezone": timezone,
            },
        )
        marine_json = marine_resp.json()

    all_times: list[str] = (marine_json.get("hourly", {}).get("time") or [])
    if not all_times:
        raise HTTPException(502, "Upstream returned no hourly time series.")

    start_index = find_start_index(all_times, timezone)
    end_index = start_index + hours

    times = all_times[start_index:end_index]
    total_wave = (marine_json["hourly"].get("wave_height") or [])[start_index:end_index]
    swell = (marine_json["hourly"].get("swell_wave_height") or [])[start_index:end_index]
    period = (marine_json["hourly"].get("swell_wave_period") or [])[start_index:end_index]
    wind_wave = (marine_json["hourly"].get("wind_wave_height") or [])[start_index:end_index]

    n = min(len(times), len(total_wave), len(swell), len(period), len(wind_wave))
    if n == 0:
        raise HTTPException(502, "Upstream returned insufficient hourly data.")

    hourly_data = []
    for i in range(n):
        est = surf_estimate_ft(swell[i], period[i], wind_wave[i])
        hourly_data.append({
            "time": times[i],
            "total_wave_m": total_wave[i],
            "swell_m": swell[i],
            "period_s": period[i],
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
                round(min(h["surf_est_ft"] for h in block), 1),
                round(max(h["surf_est_ft"] for h in block), 1),
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
            "Wind temporarily removed for stability (marine-only model).",
            "Forecast window starts at the current local hour (not midnight).",
            "Using stored lat/lon from registry.",
            "Hourly data cached per spot per hour.",
            "Surf estimate is spot-agnostic."
        ],
    }

    cache_set(cache_key, response, ttl_seconds=55 * 60)
    return response