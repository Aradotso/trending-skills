```markdown
---
name: shadowbroker-osint-platform
description: Real-time multi-domain OSINT dashboard aggregating aircraft, ships, satellites, earthquakes, CCTV, GPS jamming, and geopolitical events on a unified map interface.
triggers:
  - set up shadowbroker osint dashboard
  - track private jets and aircraft with shadowbroker
  - add a new data layer to shadowbroker
  - configure shadowbroker backend api
  - deploy shadowbroker with docker
  - extend shadowbroker with custom feeds
  - shadowbroker satellite tracking setup
  - build osint map with shadowbroker
---

# ShadowBroker OSINT Platform

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

ShadowBroker is a self-hosted, real-time geospatial intelligence dashboard built with **Next.js** (frontend), **MapLibre GL** (map rendering), and **FastAPI + Python** (backend). It aggregates public OSINT feeds — ADS-B aircraft, AIS ships, satellite TLEs, USGS earthquakes, CCTV networks, GPS jamming, conflict events, and more — into a single dark-ops map interface.

---

## Installation

### Prerequisites
- Docker + Docker Compose **or** Podman + podman-compose
- Git

### Quick Start (Linux/macOS)

```bash
git clone https://github.com/BigBodyCobain/Shadowbroker.git
cd Shadowbroker
./compose.sh up -d
```

### Quick Start (Windows)

```bash
git clone https://github.com/BigBodyCobain/Shadowbroker.git
cd Shadowbroker
docker-compose up -d
```

Open **http://localhost:3000** in your browser.

### Force Podman engine

```bash
./compose.sh --engine podman up -d
```

### Update to latest version

```bash
git pull origin main

# Linux/macOS
./compose.sh down
./compose.sh up --build -d

# Windows
docker compose down
docker compose up --build -d
```

### Clear stale cache after update

```bash
docker compose build --no-cache
docker image prune -f
```

### View backend logs

```bash
./compose.sh logs -f backend
# or
docker compose logs -f backend
```

---

## Architecture Overview

```
Frontend (Next.js + MapLibre GL)  →  Backend (FastAPI + Python)  →  Public APIs
     :3000                                  :8000
```

- **Frontend**: TypeScript/Next.js app with MapLibre GL for WebGL map rendering, layer toggles, control panels.
- **Backend**: FastAPI server that proxies, caches, and aggregates public data feeds (OpenSky, CelesTrak, USGS, GDELT, aisstream.io, NASA FIRMS, etc.).
- **No external database required** — data is cached on disk and in memory.

### Kubernetes / Helm (Advanced)

```bash
helm repo add bjw-s-labs https://bjw-s-labs.github.io/helm-charts/
helm repo update
helm install shadowbroker ./helm/chart --create-namespace --namespace shadowbroker
```

---

## Environment Variables

Create a `.env` file in the project root (never commit secrets):

```env
# AIS vessel stream (required for maritime layer)
AISSTREAM_API_KEY=your_key_here

# NASA FIRMS fire hotspots (required for fire layer)
NASA_FIRMS_API_KEY=your_key_here

# Optional: override default ports
FRONTEND_PORT=3000
BACKEND_PORT=8000
```

Reference in code as `process.env.AISSTREAM_API_KEY` (frontend) or `os.environ["AISSTREAM_API_KEY"]` (backend).

---

## Key Data Layers & Sources

| Layer | Source | Notes |
|---|---|---|
| Commercial flights | OpenSky Network | ~5,000+ aircraft, no key needed |
| Private/military jets | adsb.lol | Includes owner identification |
| Maritime AIS | aisstream.io WebSocket | 25,000+ vessels, API key required |
| Satellites | CelesTrak TLE + SGP4 | 2,000+ active, no key needed |
| Earthquakes | USGS real-time feed | 24h window |
| Fire hotspots | NASA FIRMS VIIRS | API key required |
| Conflict events | GDELT | Last 8 hours |
| Ukraine frontline | DeepState Map GeoJSON | Live |
| CCTV cameras | TfL, TxDOT, NYC DOT, Singapore LTA | No key needed |
| GPS jamming | ADS-B NAC-P analysis | Computed from flight data |
| SDR receivers | KiwiSDR network | 500+ nodes |
| Satellite imagery | NASA GIBS MODIS / Esri / Sentinel-2 | No key needed |

---

## Backend API (FastAPI) — Key Endpoints

Base URL: `http://localhost:8000`

```
GET  /api/flights/commercial       # OpenSky commercial aircraft
GET  /api/flights/private          # Private/GA aircraft
GET  /api/flights/military         # Military aircraft
GET  /api/flights/private-jets     # HNW individual jets with owner info
GET  /api/ships                    # AIS vessel positions
GET  /api/satellites               # Real-time satellite positions (SGP4)
GET  /api/earthquakes              # USGS 24h earthquake feed
GET  /api/fires                    # NASA FIRMS hotspots
GET  /api/gdelt/conflict           # GDELT conflict events
GET  /api/carriers                 # US Navy carrier strike group positions
GET  /api/cctv                     # Aggregated CCTV camera feeds
GET  /api/gps-jamming              # Computed GPS jamming zones
GET  /api/space-weather            # NOAA geomagnetic Kp index
GET  /api/internet-outages         # Georgia Tech IODA outage data
GET  /api/news                     # SIGINT/OSINT RSS aggregation
GET  /api/region-dossier?lat=&lon= # Country profile + head of state
GET  /api/sentinel?lat=&lon=       # Sentinel-2 latest image for coordinates
```

---

## Frontend Usage — Adding a Custom Layer (TypeScript)

The frontend communicates with the backend via `fetch`. Example pattern for a new data layer:

```typescript
// src/hooks/useCustomLayer.ts
import { useEffect, useState } from "react";

interface CustomFeature {
  id: string;
  lat: number;
  lon: number;
  label: string;
}

export function useCustomLayer(enabled: boolean) {
  const [features, setFeatures] = useState<CustomFeature[]>([]);

  useEffect(() => {
    if (!enabled) return;

    const fetchData = async () => {
      const res = await fetch("http://localhost:8000/api/your-endpoint");
      const data = await res.json();
      setFeatures(data.features ?? []);
    };

    fetchData();
    const interval = setInterval(fetchData, 30_000); // refresh every 30s
    return () => clearInterval(interval);
  }, [enabled]);

  return features;
}
```

### Adding the layer to MapLibre GL

```typescript
// Inside your map component
import maplibregl from "maplibre-gl";

function addCustomLayer(map: maplibregl.Map, features: CustomFeature[]) {
  const sourceId = "custom-layer";
  const layerId = "custom-layer-points";

  const geojson: GeoJSON.FeatureCollection = {
    type: "FeatureCollection",
    features: features.map((f) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [f.lon, f.lat] },
      properties: { id: f.id, label: f.label },
    })),
  };

  if (map.getSource(sourceId)) {
    (map.getSource(sourceId) as maplibregl.GeoJSONSource).setData(geojson);
  } else {
    map.addSource(sourceId, { type: "geojson", data: geojson });
    map.addLayer({
      id: layerId,
      type: "circle",
      source: sourceId,
      paint: {
        "circle-radius": 6,
        "circle-color": "#00ff88",
        "circle-opacity": 0.85,
      },
    });
  }
}
```

---

## Backend — Adding a Custom FastAPI Endpoint (Python)

```python
# backend/routers/custom.py
from fastapi import APIRouter
import httpx
import asyncio
from functools import lru_cache
import time

router = APIRouter()

_cache: dict = {"data": None, "ts": 0}
CACHE_TTL = 60  # seconds

async def fetch_source_data() -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get("https://example-public-api.org/data.json")
        resp.raise_for_status()
        return resp.json()

@router.get("/api/custom-feed")
async def custom_feed():
    now = time.time()
    if _cache["data"] is None or now - _cache["ts"] > CACHE_TTL:
        raw = await fetch_source_data()
        _cache["data"] = [
            {"id": item["id"], "lat": item["latitude"], "lon": item["longitude"], "label": item["name"]}
            for item in raw
        ]
        _cache["ts"] = now
    return {"features": _cache["data"], "count": len(_cache["data"])}
```

Register in `main.py`:

```python
from routers import custom
app.include_router(custom.router)
```

---

## Common Patterns

### WebSocket consumer for AIS ships (TypeScript)

```typescript
// Mirrors the pattern used for aisstream.io
function connectAIS(apiKey: string, onVessel: (v: Vessel) => void) {
  const ws = new WebSocket("wss://stream.aisstream.io/v0/stream");

  ws.onopen = () => {
    ws.send(JSON.stringify({
      APIkey: apiKey,
      BoundingBoxes: [[[-180, -90], [180, 90]]],
    }));
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.MessageType === "PositionReport") {
      const pos = msg.Message.PositionReport;
      onVessel({
        mmsi: pos.UserID,
        lat: pos.Latitude,
        lon: pos.Longitude,
        heading: pos.TrueHeading,
        speed: pos.Sog,
      });
    }
  };

  ws.onerror = console.error;
  return ws;
}
```

### Satellite SGP4 propagation (Python, mirrors backend pattern)

```python
from skyfield.api import load, EarthSatellite
from datetime import datetime, timezone

def get_satellite_position(tle_line1: str, tle_line2: str) -> dict:
    ts = load.timescale()
    sat = EarthSatellite(tle_line1, tle_line2, "SAT", ts)
    t = ts.now()
    geocentric = sat.at(t)
    subpoint = geocentric.subpoint()
    return {
        "lat": subpoint.latitude.degrees,
        "lon": subpoint.longitude.degrees,
        "alt_km": subpoint.elevation.km,
    }
```

### Region dossier (right-click anywhere)

```typescript
map.on("contextmenu", async (e) => {
  const { lat, lng } = e.lngLat;
  const res = await fetch(
    `http://localhost:8000/api/region-dossier?lat=${lat}&lon=${lng}`
  );
  const dossier = await res.json();
  // dossier: { country, capital, population, languages, head_of_state, summary, thumbnail }
  showDossierPanel(dossier);
});
```

### LOCATE bar — fly to coordinates or place name

```typescript
async function locate(query: string, map: maplibregl.Map) {
  // Try coordinate parse first: "31.8, 34.8"
  const coordMatch = query.match(/^(-?\d+\.?\d*),\s*(-?\d+\.?\d*)$/);
  if (coordMatch) {
    map.flyTo({ center: [parseFloat(coordMatch[2]), parseFloat(coordMatch[1])], zoom: 10 });
    return;
  }
  // Geocode via Nominatim
  const res = await fetch(
    `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1`
  );
  const [result] = await res.json();
  if (result) {
    map.flyTo({ center: [parseFloat(result.lon), parseFloat(result.lat)], zoom: 9 });
  }
}
```

### Custom RSS feed injection (SIGINT news panel)

The news feed supports up to 20 custom RSS sources with priority weights 1–5. Add sources via the UI panel or directly in backend config:

```python
# backend/config/news_feeds.py
CUSTOM_FEEDS = [
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "priority": 3},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "priority": 2},
    {"url": "https://www.bellingcat.com/feed/", "priority": 5},  # OSINT priority
]
```

---

## Map Style Presets

Cycle through visual modes via the **STYLE** button:

| Preset | Description |
|---|---|
| `DEFAULT` | Dark ops base map |
| `SATELLITE` | Esri high-res imagery |
| `FLIR` | Thermal/infrared palette |
| `NVG` | Night vision green |
| `CRT` | Retro scan-line effect |

---

## Troubleshooting

### Dashboard shows stale data after update
```bash
docker compose build --no-cache
docker image prune -f
./compose.sh up -d
```

### Backend container fails to start
```bash
docker compose logs -f backend
# Common causes: missing API keys in .env, port 8000 already in use
```

### AIS ships not appearing
- Verify `AISSTREAM_API_KEY` is set in `.env`
- aisstream.io free tier has connection limits; check their dashboard

### Aircraft layer empty
- OpenSky Network rate-limits unauthenticated requests; wait 10–15s and refresh
- Check OpenSky status at https://opensky-network.org/

### Satellite positions incorrect
- TLE data expires; the backend auto-fetches from CelesTrak. Force refresh: `docker compose restart backend`

### Port conflict
```bash
# Change ports in .env
FRONTEND_PORT=3001
BACKEND_PORT=8001
```

### Windows: `compose.sh` not found
Use `docker compose` directly (no `./compose.sh` needed on Windows CMD/PowerShell).

### Kubernetes pods not pulling latest image
```bash
kubectl rollout restart deployment/shadowbroker-frontend -n shadowbroker
kubectl rollout restart deployment/shadowbroker-backend -n shadowbroker
```

---

## License

AGPL-3.0 — All modifications must be open-sourced under the same license if distributed.
```
