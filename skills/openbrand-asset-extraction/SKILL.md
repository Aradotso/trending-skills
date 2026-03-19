```markdown
---
name: openbrand-asset-extraction
description: Extract brand assets (logos, colors, backdrops, brand name) from any website URL using OpenBrand
triggers:
  - extract brand assets from a website
  - get logo and colors from a URL
  - fetch brand colors from a website
  - extract favicon and brand name from a domain
  - get og image and brand assets
  - pull brand kit from a website URL
  - extract brand assets like Brand.dev
  - get brand logo colors backdrop from URL
---

# OpenBrand Asset Extraction

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

OpenBrand extracts brand assets — logos, colors, backdrop images, and brand name — from any website URL. It is an open-source alternative to Brand.dev, available as an npm package, REST API, MCP server, or agent skill.

## Installation

### npm package (no API key required)

```bash
npm add openbrand
# or
bun add openbrand
```

### MCP server for Claude Code / Cursor

```bash
# Without API key (uses local extraction)
claude mcp add --transport stdio openbrand -- npx -y openbrand-mcp

# With API key (uses hosted API)
claude mcp add --transport stdio \
  --env OPENBRAND_API_KEY=$OPENBRAND_API_KEY \
  openbrand -- npx -y openbrand-mcp
```

Or add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "openbrand": {
      "command": "npx",
      "args": ["-y", "openbrand-mcp"],
      "env": {
        "OPENBRAND_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### Agent skill (Claude Code, Cursor, Codex, Gemini CLI)

```bash
npx skills add ethanjyx/openbrand
```

### Self-host the web app

```bash
git clone https://github.com/ethanjyx/openbrand.git
cd openbrand
bun install
bun dev
# Open http://localhost:3000
```

No environment variables required for local dev.

## npm Package Usage

### Basic extraction

```typescript
import { extractBrandAssets } from "openbrand";

const result = await extractBrandAssets("https://stripe.com");

if (result.ok) {
  const { brand_name, logos, colors, backdrop_images } = result.data;
  console.log("Brand name:", brand_name);         // "Stripe"
  console.log("Logos:", logos);                    // LogoAsset[]
  console.log("Colors:", colors);                  // ColorAsset[]
  console.log("Backdrops:", backdrop_images);      // BackdropAsset[]
} else {
  console.error("Error code:", result.error.code);       // "ACCESS_BLOCKED" | "NOT_FOUND" | "SERVER_ERROR" | ...
  console.error("Error message:", result.error.message); // human-readable explanation
}
```

### Working with logos

```typescript
import { extractBrandAssets } from "openbrand";

const result = await extractBrandAssets("https://github.com");

if (result.ok) {
  const { logos } = result.data;

  // Find the highest-resolution logo
  const best = logos.reduce((a, b) =>
    (a.width ?? 0) * (a.height ?? 0) >= (b.width ?? 0) * (b.height ?? 0) ? a : b
  );

  console.log("Best logo URL:", best.url);
  console.log("Dimensions:", best.width, "x", best.height);
  console.log("Format:", best.format); // "svg" | "png" | "ico" | ...
  console.log("Source:", best.source); // "favicon" | "apple-touch-icon" | "nav-logo" | "inline-svg"
}
```

### Working with colors

```typescript
import { extractBrandAssets } from "openbrand";

const result = await extractBrandAssets("https://linear.app");

if (result.ok) {
  const { colors } = result.data;

  colors.forEach((color) => {
    console.log("Hex:", color.hex);        // "#5E6AD2"
    console.log("Source:", color.source);  // "theme-color" | "manifest" | "logo-dominant"
  });

  // Get the primary brand color
  const primary = colors.find((c) => c.source === "theme-color") ?? colors[0];
  console.log("Primary color:", primary?.hex);
}
```

### Working with backdrop images

```typescript
import { extractBrandAssets } from "openbrand";

const result = await extractBrandAssets("https://vercel.com");

if (result.ok) {
  const { backdrop_images } = result.data;

  backdrop_images.forEach((img) => {
    console.log("URL:", img.url);
    console.log("Source:", img.source); // "og:image" | "css-background" | "hero-image"
    console.log("Alt:", img.alt);
  });

  // Get the OG image (best for social previews)
  const ogImage = backdrop_images.find((img) => img.source === "og:image");
}
```

### Error handling patterns

```typescript
import { extractBrandAssets } from "openbrand";

async function getBrandSafely(url: string) {
  const result = await extractBrandAssets(url);

  if (!result.ok) {
    switch (result.error.code) {
      case "ACCESS_BLOCKED":
        // Site blocks scrapers — try a different approach or skip
        console.warn("Site blocks automated access:", url);
        return null;
      case "NOT_FOUND":
        console.warn("URL not found:", url);
        return null;
      case "SERVER_ERROR":
        console.error("Target server error:", result.error.message);
        return null;
      default:
        console.error("Unknown error:", result.error.message);
        return null;
    }
  }

  return result.data;
}
```

### Batch extraction

```typescript
import { extractBrandAssets } from "openbrand";

const urls = [
  "https://stripe.com",
  "https://github.com",
  "https://vercel.com",
  "https://linear.app",
];

// Parallel extraction with error resilience
const results = await Promise.allSettled(
  urls.map((url) => extractBrandAssets(url))
);

const brands = results
  .map((r, i) => ({ url: urls[i], result: r }))
  .filter(({ result }) => result.status === "fulfilled" && (result as PromiseFulfilledResult<any>).value.ok)
  .map(({ url, result }) => ({
    url,
    ...(result as PromiseFulfilledResult<any>).value.data,
  }));

console.log("Successfully extracted:", brands.length, "brands");
```

## REST API Usage

Get a free API key at [openbrand.sh/dashboard](https://openbrand.sh/dashboard).

### cURL

```bash
curl "https://openbrand.sh/api/extract?url=https://stripe.com" \
  -H "Authorization: Bearer $OPENBRAND_API_KEY"
```

### TypeScript fetch

```typescript
async function extractBrand(url: string) {
  const res = await fetch(
    `https://openbrand.sh/api/extract?url=${encodeURIComponent(url)}`,
    {
      headers: {
        Authorization: `Bearer ${process.env.OPENBRAND_API_KEY}`,
      },
    }
  );

  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

const brand = await extractBrand("https://stripe.com");
console.log(brand.brand_name); // "Stripe"
```

### Python requests

```python
import requests
import os

def extract_brand(url: str) -> dict:
    res = requests.get(
        "https://openbrand.sh/api/extract",
        params={"url": url},
        headers={"Authorization": f"Bearer {os.environ['OPENBRAND_API_KEY']}"},
    )
    res.raise_for_status()
    return res.json()

brand = extract_brand("https://stripe.com")
print(brand["brand_name"])   # "Stripe"
print(brand["colors"])       # list of color assets
```

## What Gets Extracted

| Asset | Sources |
|---|---|
| **Logos** | favicons, apple-touch-icons, header/nav logos, inline SVGs (with dimension probing) |
| **Brand colors** | `theme-color` meta tag, `manifest.json`, dominant colors from logo imagery |
| **Backdrop images** | `og:image`, CSS backgrounds, hero/banner images |
| **Brand name** | `og:site_name`, `application-name`, logo alt text, page `<title>` |

## Common Patterns

### Building a brand card component

```typescript
import { extractBrandAssets } from "openbrand";

interface BrandCard {
  name: string;
  logoUrl: string | null;
  primaryColor: string | null;
  ogImage: string | null;
}

async function buildBrandCard(url: string): Promise<BrandCard | null> {
  const result = await extractBrandAssets(url);
  if (!result.ok) return null;

  const { brand_name, logos, colors, backdrop_images } = result.data;

  // Prefer SVG, then largest raster
  const logo =
    logos.find((l) => l.format === "svg") ??
    logos.sort(
      (a, b) => (b.width ?? 0) * (b.height ?? 0) - (a.width ?? 0) * (a.height ?? 0)
    )[0] ??
    null;

  const primaryColor =
    colors.find((c) => c.source === "theme-color")?.hex ??
    colors[0]?.hex ??
    null;

  const ogImage =
    backdrop_images.find((img) => img.source === "og:image")?.url ?? null;

  return {
    name: brand_name ?? new URL(url).hostname,
    logoUrl: logo?.url ?? null,
    primaryColor,
    ogImage,
  };
}
```

### Enriching a list of companies

```typescript
import { extractBrandAssets } from "openbrand";

const companies = [
  { name: "Stripe", website: "https://stripe.com" },
  { name: "Vercel", website: "https://vercel.com" },
];

const enriched = await Promise.all(
  companies.map(async (company) => {
    const result = await extractBrandAssets(company.website);
    if (!result.ok) return { ...company, brand: null };

    return {
      ...company,
      brand: {
        logo: result.data.logos[0]?.url ?? null,
        color: result.data.colors[0]?.hex ?? null,
      },
    };
  })
);
```

## Tech Stack (for self-hosting contributors)

- **Framework**: Next.js + React
- **Language**: TypeScript
- **HTML parsing**: Cheerio
- **Image processing**: Sharp
- **Styling**: Tailwind CSS
- **Runtime**: Bun (recommended) or Node.js

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ACCESS_BLOCKED` error | Site blocks scrapers (Cloudflare, etc.) | Use the hosted API which has bypass strategies; nothing the npm package can do |
| Empty `logos` array | Site uses JS-rendered logos | Consider the hosted API which may handle JS rendering |
| `NOT_FOUND` error | Invalid URL or domain doesn't resolve | Verify the URL is reachable in a browser |
| Colors array is empty | No `theme-color` meta, no manifest, no logo to analyze | Expected for minimal sites |
| MCP tool not appearing | MCP server not installed | Run `claude mcp list` to verify installation |

## Links

- **Web app**: [openbrand.sh](https://openbrand.sh)
- **API dashboard**: [openbrand.sh/dashboard](https://openbrand.sh/dashboard)
- **npm package**: [npmjs.com/package/openbrand](https://www.npmjs.com/package/openbrand)
- **MCP package**: [npmjs.com/package/openbrand-mcp](https://www.npmjs.com/package/openbrand-mcp)
- **Agent skill**: [skills.sh/ethanjyx/openbrand](https://skills.sh/ethanjyx/openbrand/openbrand)
- **GitHub**: [github.com/ethanjyx/openbrand](https://github.com/ethanjyx/openbrand)
- **License**: MIT
```
