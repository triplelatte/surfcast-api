from __future__ import annotations

import os
import csv
import io
import time
import math
import hashlib
import asyncio
from typing import Any, Optional, Literal
from datetime import datetime, date, timedelta
from urllib.parse import quote

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
# Config
# -----------------------------
OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"
SPOTS_CSV_PATH = os.getenv("SPOTS_CSV_PATH", "./surf_registry_us_v4_with_latlon.csv")
BASE_URL = os.getenv("BASE_URL", "https://surfcast-api.onrender.com").rstrip("/")

FORECAST_TTL_SECONDS = 55 * 60
PLOT_TTL_SECONDS = 55 * 60
SUN_TTL_SECONDS = 26 * 60 * 60


# -----------------------------
# Simple in-memory TTL caches
# -----------------------------
_FORECAST_CACHE: dict[str, tuple[float, Any]] = {}  # forecast JSON
_PLOT_CACHE: dict[str, tuple[float, Any]] = {}  # image bytes
_SUN_CACHE: dict[str, tuple[float, Any]] = {}  # (sunrise, sunset)


def _cache_get(store: dict[str, tuple[float, Any]], key: str) -> Optional[Any]:
    item = store.get(key)
    if not item:
        return None
    exp, val = item
    if time.time() > exp:
        store.pop(key, None)
        return None
    return val


def _cache_set(store: dict[str, tuple[float, Any]], key: str, val: Any, ttl_seconds: int) -> None:
    store[key] = (time.time() + ttl_seconds, val)


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
    plot_url: str
    plot_url_small: str


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="SurfCast API", version="0.9.0")


# -----------------------------
# Spot registry (CSV, no pandas)
# -----------------------------
def _load_spots_csv(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise RuntimeError(f"SPOTS_CSV_PATH not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"spot_id", "spot_name", "state", "region", "lat", "lon"}
        headers = set(reader.fieldnames or [])
        if not required.issubset(headers):
            raise RuntimeError(
                f"CSV missing required columns. Need at least: {sorted(required)}; found: {sorted(headers)}"
            )

        for r in reader:
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
                detail="Upstream rate limited (429) from Open-Meteo marine endpoint.",
            )

        await asyncio.sleep(delay)
        delay *= 2.0


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
    Cached sunrise/sunset per (lat,lon,date,tz).
    """
    key = stable_key("sun", f"{lat:.4f}", f"{lon:.4f}", str(tz), d.isoformat())
    cached = _cache_get(_SUN_CACHE, key)
    if cached:
        return cached

    obs = Observer(latitude=lat, longitude=lon)
    s = sun(obs, date=d, tzinfo=tz)
    bounds = (s["sunrise"], s["sunset"])
    _cache_set(_SUN_CACHE, key, bounds, ttl_seconds=SUN_TTL_SECONDS)
    return bounds


def is_daylight(dt_local: datetime, lat: float, lon: float, tz) -> bool:
    sunrise, sunset = daylight_bounds_for_date(dt_local.date(), lat, lon, tz)
    return sunrise <= dt_local <= sunset


# -----------------------------
# URL helpers (for GPT embedding)
# -----------------------------
def _tz_q(timezone: str) -> str:
    return quote(timezone, safe="")


def build_plot_url(
    spot_id: str,
    hours: int,
    timezone: str,
    *,
    fmt: str = "png",
    width: int = 1200,
    height: int = 500,
    dpi: int = 140,
) -> str:
    # include timezone; safely encoded
    return (
        f"{BASE_URL}/v1/forecast_plot?"
        f"spot_id={spot_id}&hours={hours}&timezone={_tz_q(timezone)}"
        f"&format={fmt}&width={width}&height={height}&dpi={dpi}"
    )


def build_plot_url_small_jpg(spot_id: str, hours: int, timezone: str) -> str:
    # Smaller + JPEG: more likely to render in GPT UIs
    return build_plot_url(
        spot_id=spot_id,
        hours=hours,
        timezone=timezone,
        fmt="jpg",
        width=900,
        height=360,
        dpi=110,
    )


# -----------------------------
# Matplotlib renderer
# -----------------------------
def render_forecast_image(
    *,
    spot_name: str,
    tz,
    hourly: list[dict[str, Any]],
    best_windows: list[dict[str, Any]],
    lat: float,
    lon: float,
    width: int,
    height: int,
    dpi: int,
    fmt: Literal["png", "jpg"],
) -> bytes:
    # Parse series
    x: list[datetime] = []
    y: list[float] = []

    for h in hourly:
        try:
            dt = datetime.fromisoformat(h["time"]).replace(tzinfo=tz)
        except Exception:
            continue
        x.append(dt)

        val = h.get("total_wave_m")
        try:
            y.append(float(val) if val is not None else float("nan"))
        except Exception:
            y.append(float("nan"))

    if not x:
        raise HTTPException(502, "No hourly data to plot")

    # Daylight shading per day
    dates = sorted({dt.date() for dt in x})
    day_bounds = {d: daylight_bounds_for_date(d, lat, lon, tz) for d in dates}

    # Best window spans
    best_spans: list[tuple[datetime, datetime]] = []
    for bw in best_windows:
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

    # Highlight best windows
    for s, e in best_spans:
        ax.axvspan(s, e, alpha=0.20)

    ax.set_title(f"{spot_name} — Total Wave (m)")
    ax.set_ylabel("total_wave_m")
    ax.set_xlabel("Local time")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    if fmt == "jpg":
        # JPEG is much smaller and more likely to embed reliably
        fig.savefig(buf, format="jpg", dpi=dpi, quality=85)
    else:
        fig.savefig(buf, format="png", dpi=dpi)

    plt.close(fig)
    return buf.getvalue()


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

    hour_bucket = str(int(time.time() // 3600))
    cache_key = stable_key("forecast", spot_id, str(hours), timezone, hour_bucket)
    cached = _cache_get(_FORECAST_CACHE, cache_key)
    if cached:
        return cached

    hourly_vars = ",".join(
        [
            "wave_height,wave_period,wave_direction",
            "swell_wave_height,swell_wave_period,swell_wave_direction",
            "wind_wave_height",
        ]
    )

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
        hourly_data.append(
            {
                "time": times[i],
                "total_wave_m": total_wave[i],
                "swell_m": swell[i],
                "period_s": period[i],
                "wind_wave_m": wind_wave[i],
                "surf_est_ft": est,
            }
        )

    # ---- Daylight-only ranking for best windows (non-overlapping 3-hour blocks) ----
    best_windows: list[dict[str, Any]] = []
    scores: list[tuple[float, int]] = []
    tz = ZoneInfo(timezone) if ZoneInfo is not None else None

    for i in range(len(hourly_data) - 2):
        block = hourly_data[i : i + 3]

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

    chosen_ranges: list[tuple[int, int]] = []
    for avg, i in scores:
        start_idx = i
        end_idx = i + 2
        overlaps = any(not (end_idx < s or start_idx > e) for (s, e) in chosen_ranges)
        if overlaps:
            continue

        block = hourly_data[start_idx : end_idx + 1]
        block_vals = [h["surf_est_ft"] for h in block if h["surf_est_ft"] is not None]
        if not block_vals:
            continue

        best_windows.append(
            {
                "start": block[0]["time"],
                "end": block[-1]["time"],
                "surf_est_ft_range": [round(min(block_vals), 1), round(max(block_vals), 1)],
                "summary": "Top 3-hour surf window (daylight only; non-overlapping).",
            }
        )
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
        "notes": [
            "Marine-only forecast via Open-Meteo marine endpoint.",
            "Forecast window starts at the current local hour (not midnight).",
            "Best windows are restricted to daylight (sunrise→sunset).",
            "Best windows are non-overlapping 3-hour blocks.",
            "Using stored lat/lon from registry.",
            "Hourly data cached per spot per hour (55 min TTL).",
            "Surf estimate is spot-agnostic and may vary due to local bathymetry and sandbars.",
        ],
        # Provide fully-formed URLs so GPT never assembles incorrectly.
        "plot_url": build_plot_url(
            spot_id=spot_id, hours=hours, timezone=timezone, fmt="png", width=1200, height=500, dpi=140
        ),
        "plot_url_small": build_plot_url_small_jpg(spot_id=spot_id, hours=hours, timezone=timezone),
    }

    _cache_set(_FORECAST_CACHE, cache_key, response, ttl_seconds=FORECAST_TTL_SECONDS)
    return response


# -----------------------------
# API: forecast plot (PNG/JPG)
# -----------------------------
@app.get("/v1/forecast_plot")
async def forecast_plot(
    spot_id: str,
    hours: int = Query(72, ge=6, le=168),
    timezone: str = "America/Los_Angeles",
    format: str = Query("jpg", pattern="^(png|jpg)$"),
    # Smaller defaults are more likely to render in GPT UIs
    width: int = Query(900, ge=600, le=2400),
    height: int = Query(360, ge=300, le=1200),
    dpi: int = Query(110, ge=90, le=200),
):
    hour_bucket = str(int(time.time() // 3600))
    plot_key = stable_key(
        "plot",
        spot_id,
        str(hours),
        timezone,
        hour_bucket,
        format,
        str(width),
        str(height),
        str(dpi),
    )

    cached_img = _cache_get(_PLOT_CACHE, plot_key)
    if cached_img:
        mime = "image/png" if format == "png" else "image/jpeg"
        return Response(
            content=cached_img,
            media_type=mime,
            headers={
                "Content-Disposition": f"inline; filename=surfcast.{format}",
                "Cache-Control": f"public, max-age={PLOT_TTL_SECONDS}",
            },
        )

    fc = await forecast(spot_id=spot_id, hours=hours, timezone=timezone)

    if ZoneInfo is None:
        raise HTTPException(500, "zoneinfo not available on this runtime")
    tz = ZoneInfo(timezone)

    lat = float(fc["spot"]["lat"])
    lon = float(fc["spot"]["lon"])
    spot_name = fc["spot"].get("name", spot_id)

    fmt = "png" if format == "png" else "jpg"
    img = render_forecast_image(
        spot_name=spot_name,
        tz=tz,
        hourly=fc["hourly"],
        best_windows=fc["best_windows"],
        lat=lat,
        lon=lon,
        width=width,
        height=height,
        dpi=dpi,
        fmt=fmt,  # type: ignore[arg-type]
    )

    _cache_set(_PLOT_CACHE, plot_key, img, ttl_seconds=PLOT_TTL_SECONDS)

    mime = "image/png" if format == "png" else "image/jpeg"
    return Response(
        content=img,
        media_type=mime,
        headers={
            "Content-Disposition": f"inline; filename=surfcast.{format}",
            "Cache-Control": f"public, max-age={PLOT_TTL_SECONDS}",
        },
    )