from __future__ import annotations

import os
import csv
import io
import time
import math
import hashlib
import asyncio
from typing import Any, Optional
from datetime import datetime, date, timedelta

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from astral import Observer
from astral.sun import sun

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Simple in-memory TTL caches
# -----------------------------
_CACHE: dict[str, tuple[float, Any]] = {}        # forecast JSON
_PLOT_CACHE: dict[str, tuple[float, Any]] = {}   # PNG bytes
_SUN_CACHE: dict[str, tuple[float, Any]] = {}    # (sunrise, sunset)

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

def cache_get_from(store: dict[str, tuple[float, Any]], key: str) -> Optional[Any]:
    item = store.get(key)
    if not item:
        return None
    exp, val = item
    if time.time() > exp:
        store.pop(key, None)
        return None
    return val

def cache_set_to(store: dict[str, tuple[float, Any]], key: str, val: Any, ttl_seconds: int) -> None:
    store[key] = (time.time() + ttl_seconds, val)

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


app = FastAPI(title="SurfCast API", version="0.7.0")


# -----------------------------
# Spot registry (CSV, no pandas)
# -----------------------------
SPOTS_CSV_PATH = os.getenv("SPOTS_CSV_PATH", "./surf_registry_us_v4_with_latlon.csv")

def _load_spots_csv(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise RuntimeError(f"SPOTS_CSV_PATH not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"spot_id", "spot_name", "state", "region", "lat", "lon"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                f"CSV missing required columns. Need at least: {sorted(required)}; "
                f"found: {reader.fieldnames}"
            )
        for r in reader:
            # Keep everything as strings except lat/lon
            try:
                r["lat"] = float(r["lat"])
                r["lon"] = float(r["lon"])
            except Exception:
                continue
            rows.append(r)
    return rows

spots_rows: list[dict[str, Any]] = _load_spots_csv(SPOTS_CSV_PATH)
spots_by_id: dict[str, dict[str, Any]] = {r["spot_id"]: r for r in spots_rows}


@app.get("/health")
def health():
    return {"status": "ok", "spots_loaded": int(len(spots_rows))}


@app.get("/v1/spots/search")
def search_spots(q: str = Query(..., min_length=2), limit: int = 20):
    qn = q.strip().lower()
    out: list[dict[str, Any]] = []
    for r in spots_rows:
        name = str(r.get("spot_name", "")).lower()
        if qn in name:
            out.append(
                {
                    "spot_id": r.get("spot_id"),
                    "spot_name": r.get("spot_name"),
                    "state": r.get("state"),
                    "region": r.get("region"),
                }
            )
            if len(out) >= limit:
                break
    return out


# -----------------------------
# HTTP helper: backoff on 429
# -----------------------------
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


# -----------------------------
# Surf scoring + time alignment
# -----------------------------
def surf_estimate_ft(swell_m, period_s, wind_wave_m):
    if swell_m is None or period_s is None:
        return None

    energy = float(swell_m) * (max(float(period_s), 1.0) ** 0.5)
    chop = float(wind_wave_m or 0.0)
    face_ft = 3.5 * energy - 2.0 * chop
    return max(face_ft, 0.0)


def find_start_index(times: list[str], timezone: str) -> int:
    """
    Open-Meteo times are local strings like 'YYYY-MM-DDTHH:MM' in the requested timezone.
    Start at the first timestamp >= the current local hour.
    """
    if ZoneInfo is None:
        return 0

    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    now_floor = now.replace(minute=0, second=0, microsecond=0)

    now_key = now_floor.strftime("%Y-%m-%dT%H:%M")
    try:
        return times.index(now_key)
    except ValueError:
        pass

    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t).replace(tzinfo=tz)
        except ValueError:
            continue
        if dt >= now_floor:
            return i

    return 0


# -----------------------------
# Daylight helpers (cached)
# -----------------------------
def daylight_bounds_for_date(d: date, lat: float, lon: float, tz) -> tuple[datetime, datetime]:
    """
    Returns (sunrise, sunset) for the date at lat/lon in tz.
    Cached so Astral isn't called repeatedly.
    """
    key = stable_key("sun", f"{lat:.4f}", f"{lon:.4f}", str(tz), d.isoformat())
    cached = cache_get_from(_SUN_CACHE, key)
    if cached:
        return cached

    obs = Observer(latitude=lat, longitude=lon)
    s = sun(obs, date=d, tzinfo=tz)
    bounds = (s["sunrise"], s["sunset"])
    cache_set_to(_SUN_CACHE, key, bounds, ttl_seconds=26 * 60 * 60)
    return bounds


def is_daylight(dt_local: datetime, lat: float, lon: float, tz) -> bool:
    sunrise, sunset = daylight_bounds_for_date(dt_local.date(), lat, lon, tz)
    return sunrise <= dt_local <= sunset


# -----------------------------
# API: forecast (JSON)
# -----------------------------
@app.get("/v1/forecast", response_model=ForecastResponse)
async def forecast(
    spot_id: str,
    hours: int = Query(72, ge=6, le=168),
    timezone: str = "America/Los_Angeles",
):
    spot = spots_by_id.get(spot_id)
    if not spot:
        raise HTTPException(404, "Spot not found")

    lat = float(spot["lat"])
    lon = float(spot["lon"])

    # Cache per hour bucket (keeps upstream calls low)
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

    hourly_data: list[dict[str, Any]] = []
    for i in range(n):
        est = surf_estimate_ft(swell[i], period[i], wind_wave[i])
        hourly_data.append({
            "time": times[i],
            "total_wave_m": total_wave[i],
            "swell_m": swell[i],
            "period_s": period[i],
            "wind_wave_m": wind_wave[i],
            "surf_est_ft": est,
        })

    # ---- Daylight-only ranking for best windows (non-overlapping 3-hour blocks) ----
    best_windows: list[dict[str, Any]] = []
    scores: list[tuple[float, int]] = []

    if ZoneInfo is None:
        tz = None
    else:
        tz = ZoneInfo(timezone)

    for i in range(len(hourly_data) - 2):
        block = hourly_data[i:i+3]

        # Require all 3 hours to be between sunrise and sunset
        if tz is not None:
            ok = True
            for h in block:
                try:
                    dt_local = datetime.fromisoformat(h["time"]).replace(tzinfo=tz)
                except ValueError:
                    ok = False
                    break
                if not is_daylight(dt_local, lat, lon, tz):
                    ok = False
                    break
            if not ok:
                continue

        vals = [h["surf_est_ft"] for h in block if h["surf_est_ft"] is not None]
        if len(vals) < 2:
            continue
        scores.append((sum(vals) / len(vals), i))

    scores.sort(reverse=True)

    chosen_ranges: list[tuple[int, int]] = []  # inclusive indices
    for avg, i in scores:
        start_idx = i
        end_idx = i + 2  # 3 hours block

        overlaps = any(not (end_idx < s or start_idx > e) for (s, e) in chosen_ranges)
        if overlaps:
            continue

        block = hourly_data[start_idx:end_idx + 1]
        best_windows.append({
            "start": block[0]["time"],
            "end": block[-1]["time"],
            "surf_est_ft_range": [
                round(min(h["surf_est_ft"] for h in block if h["surf_est_ft"] is not None), 1),
                round(max(h["surf_est_ft"] for h in block if h["surf_est_ft"] is not None), 1),
            ],
            "summary": "Top 3-hour surf window (daylight only; non-overlapping).",
        })
        chosen_ranges.append((start_idx, end_idx))

        if len(best_windows) >= 3:
            break

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
        "plot_url": f"https://surfcast-api.onrender.com/v1/forecast_plot?spot_id={spot_id}&hours={hours}&timezone={timezone}",
        "notes": [
            "Marine-only forecast via Open-Meteo marine endpoint.",
            "Forecast window starts at the current local hour (not midnight).",
            "Best windows are restricted to daylight (sunrise→sunset).",
            "Best windows are non-overlapping 3-hour blocks.",
            "Using stored lat/lon from registry.",
            "Hourly data cached per spot per hour (55 min TTL).",
        ],
    }

    cache_set(cache_key, response, ttl_seconds=55 * 60)
    return response


# -----------------------------
# API: forecast plot (PNG)
# -----------------------------
@app.get("/v1/forecast_plot")
async def forecast_plot(
    spot_id: str,
    hours: int = Query(72, ge=6, le=168),
    timezone: str = "America/Los_Angeles",
    width: int = Query(1200, ge=600, le=2400),
    height: int = Query(500, ge=300, le=1200),
    dpi: int = Query(140, ge=90, le=200),
):
    # Align plot cache to same hour bucket as forecast cache
    hour_bucket = str(int(time.time() // 3600))
    plot_key = stable_key(
        "plot", spot_id, str(hours), timezone, hour_bucket, str(width), str(height), str(dpi)
    )

    cached_png = cache_get_from(_PLOT_CACHE, plot_key)
    if cached_png:
        return Response(content=cached_png, media_type="image/png")

    # Reuse forecast logic (will hit forecast cache if warm)
    fc = await forecast(spot_id=spot_id, hours=hours, timezone=timezone)

    if ZoneInfo is None:
        raise HTTPException(500, "zoneinfo not available on this runtime")

    tz = ZoneInfo(timezone)
    lat = float(fc["spot"]["lat"])
    lon = float(fc["spot"]["lon"])

    # Parse times + values
    x: list[datetime] = []
    y: list[float] = []
    for h in fc["hourly"]:
        try:
            dt = datetime.fromisoformat(h["time"]).replace(tzinfo=tz)
        except ValueError:
            continue
        x.append(dt)
        val = h.get("total_wave_m")
        try:
            y.append(float(val) if val is not None else float("nan"))
        except Exception:
            y.append(float("nan"))

    if not x:
        raise HTTPException(502, "No hourly data to plot")

    # Daylight shading spans per day (handles multi-day forecasts)
    dates = sorted({dt.date() for dt in x})
    day_bounds = {d: daylight_bounds_for_date(d, lat, lon, tz) for d in dates}

    # Best windows (green spans)
    best_spans: list[tuple[datetime, datetime]] = []
    for bw in fc.get("best_windows", []):
        try:
            s = datetime.fromisoformat(bw["start"]).replace(tzinfo=tz)
            e = datetime.fromisoformat(bw["end"]).replace(tzinfo=tz) + timedelta(hours=1)
            best_spans.append((s, e))
        except Exception:
            continue

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.plot(x, y)

    # Shade dusk->dawn (outside sunrise->sunset)
    tmin = x[0]
    tmax = x[-1] + timedelta(hours=1)

    for d in dates:
        sunrise, sunset = day_bounds[d]
        day_start = datetime(d.year, d.month, d.day, 0, 0, tzinfo=tz)
        day_end = day_start + timedelta(days=1)

        seg_start = max(tmin, day_start)
        seg_end = min(tmax, day_end)
        if seg_start >= seg_end:
            continue

        if seg_start < sunrise:
            ax.axvspan(seg_start, min(sunrise, seg_end), alpha=0.15)
        if sunset < seg_end:
            ax.axvspan(max(sunset, seg_start), seg_end, alpha=0.15)

    # Highlight best windows in green
    for s, e in best_spans:
        ax.axvspan(s, e, alpha=0.20)

    spot_name = fc["spot"].get("name", spot_id)
    ax.set_title(f"{spot_name} — Total Wave (m)")
    ax.set_ylabel("total_wave_m")
    ax.set_xlabel("Local time")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    png = buf.getvalue()

    cache_set_to(_PLOT_CACHE, plot_key, png, ttl_seconds=55 * 60)
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=surfcast.png",
            "Cache-Control": "public, max-age=3300",
        },