```markdown
---
name: world2agent-protocol
description: Skill for using World2Agent (W2A) — the open protocol that standardizes how AI agents perceive the real world via structured sensor signals.
triggers:
  - add a sensor to my agent
  - set up world2agent
  - how do I use W2A sensors
  - make my agent perceive real-world data
  - install a world2agent sensor
  - build a custom W2A sensor
  - configure world2agent in my project
  - stream real-time signals to my AI agent
---

# World2Agent (W2A) Protocol

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

World2Agent (W2A) is an open protocol that standardizes how AI agents perceive the real world. Sensors watch data sources and emit structured signals following the W2A schema. Your agent receives those signals and decides what to do. The architecture is always: **World → Sensor → Agent**.

---

## Installation

### Option 1: Claude Code Plugin (Fastest)

In an active Claude Code session:

```
/plugin marketplace add machinepulse-ai/world2agent-plugins
/plugin install world2agent@world2agent-plugins
/reload-plugins
```

Add sensors:

```
/world2agent:sensor-add @world2agent/sensor-hackernews
/world2agent:sensor-add @quill-io/sensor-frontier-ai-news
```

Restart Claude Code with the plugin channel loaded:

```bash
claude --dangerously-load-development-channels plugin:world2agent@world2agent-plugins
```

### Option 2: SDK / Code Integration

Install the core SDK and a sensor:

```bash
npm install @world2agent/sdk
npm install @world2agent/sensor-hackernews
```

---

## Core Concepts

| Term | Description |
|------|-------------|
| **Sensor** | An npm package that watches a data source and emits W2A signals |
| **Signal** | A structured JSON object emitted by a sensor (follows W2A schema) |
| **SensorHub** | Catalog of all official and community sensors at [world2agent.ai/hub](https://world2agent.ai/hub) |
| **Channel** | A named stream that groups signals from one or more sensors |

---

## Signal Format

Every W2A signal shares a common schema:

```typescript
interface W2ASignal {
  id: string;                  // Unique signal ID (UUID)
  sensorId: string;            // e.g. "@world2agent/sensor-hackernews"
  sensorVersion: string;       // semver
  timestamp: string;           // ISO 8601
  type: string;                // e.g. "story.new", "price.alert", "news.post"
  priority: "low" | "medium" | "high" | "critical";
  payload: Record<string, unknown>; // Sensor-specific structured data
  meta?: {
    source?: string;           // URL or origin
    tags?: string[];
    confidence?: number;       // 0–1
  };
}
```

Example signal from the Hacker News sensor:

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "sensorId": "@world2agent/sensor-hackernews",
  "sensorVersion": "1.2.0",
  "timestamp": "2026-04-29T10:00:00Z",
  "type": "story.new",
  "priority": "medium",
  "payload": {
    "storyId": 39812345,
    "title": "Show HN: My open-source project",
    "url": "https://example.com",
    "score": 142,
    "author": "someone",
    "commentCount": 38
  },
  "meta": {
    "source": "https://news.ycombinator.com/item?id=39812345",
    "tags": ["show-hn", "open-source"]
  }
}
```

---

## SDK Usage

### Subscribing to Sensor Signals

```typescript
import { W2AClient } from "@world2agent/sdk";
import { HackerNewsSensor } from "@world2agent/sensor-hackernews";

const client = new W2AClient();

// Register sensors
client.use(new HackerNewsSensor({
  filter: {
    minScore: 100,
    types: ["story.new", "story.trending"],
  },
}));

// Listen for signals
client.on("signal", (signal) => {
  console.log(`[${signal.priority}] ${signal.type}:`, signal.payload);
});

// Start receiving signals
await client.start();
```

### Handling Specific Signal Types

```typescript
import { W2AClient, W2ASignal } from "@world2agent/sdk";

const client = new W2AClient();

client.on("signal", (signal: W2ASignal) => {
  switch (signal.type) {
    case "story.new":
      handleNewStory(signal.payload);
      break;
    case "price.alert":
      handlePriceAlert(signal.payload);
      break;
    default:
      console.log("Unhandled signal type:", signal.type);
  }
});

function handleNewStory(payload: Record<string, unknown>) {
  const { title, score, url } = payload as {
    title: string;
    score: number;
    url: string;
  };
  console.log(`New story (score: ${score}): ${title} — ${url}`);
}
```

### Multiple Sensors

```typescript
import { W2AClient } from "@world2agent/sdk";
import { HackerNewsSensor } from "@world2agent/sensor-hackernews";
import { FrontierAINewsSensor } from "@quill-io/sensor-frontier-ai-news";

const client = new W2AClient({
  channelName: "my-agent-feed",
});

client.use(new HackerNewsSensor({ minScore: 50 }));
client.use(new FrontierAINewsSensor({ labs: ["openai", "anthropic", "google"] }));

client.on("signal", (signal) => {
  // Signals from all sensors arrive here with sensorId to distinguish source
  console.log(`From ${signal.sensorId}:`, signal.type, signal.payload);
});

await client.start();
```

---

## Building a Custom Sensor

Install the build skill:

```bash
npx skills add https://github.com/machinepulse-ai/world2agent/skills/build-w2a-sensor
```

Or build manually in ~50 lines:

```typescript
// src/index.ts
import { BaseSensor, W2ASignal, SensorConfig } from "@world2agent/sdk";

interface MyWeatherSensorConfig extends SensorConfig {
  city: string;
  apiKey?: string; // use process.env.WEATHER_API_KEY
  pollIntervalMs?: number;
}

export class WeatherSensor extends BaseSensor {
  private city: string;
  private apiKey: string;
  private pollInterval: number;

  constructor(config: MyWeatherSensorConfig) {
    super({
      id: "@myorg/sensor-weather",
      version: "1.0.0",
    });
    this.city = config.city;
    this.apiKey = config.apiKey ?? process.env.WEATHER_API_KEY ?? "";
    this.pollInterval = config.pollIntervalMs ?? 60_000;
  }

  async start(): Promise<void> {
    await this.poll(); // immediate first fetch
    setInterval(() => this.poll(), this.pollInterval);
  }

  private async poll(): Promise<void> {
    const res = await fetch(
      `https://api.weatherapi.com/v1/current.json?key=${this.apiKey}&q=${this.city}`
    );
    const data = await res.json();

    const signal: Omit<W2ASignal, "id" | "timestamp"> = {
      sensorId: this.meta.id,
      sensorVersion: this.meta.version,
      type: "weather.update",
      priority: data.current.temp_c > 35 ? "high" : "low",
      payload: {
        city: this.city,
        tempC: data.current.temp_c,
        condition: data.current.condition.text,
        humidity: data.current.humidity,
        windKph: data.current.wind_kph,
      },
      meta: {
        source: `https://www.weatherapi.com`,
        tags: ["weather", this.city.toLowerCase()],
      },
    };

    this.emit(signal); // BaseSensor handles id + timestamp
  }
}
```

### package.json for your sensor

```json
{
  "name": "@myorg/sensor-weather",
  "version": "1.0.0",
  "description": "W2A sensor for real-time weather data",
  "main": "dist/index.js",
  "keywords": ["w2a-sensor", "world2agent", "weather"],
  "peerDependencies": {
    "@world2agent/sdk": "^1.0.0"
  }
}
```

Publish it:

```bash
npm publish --access public
```

---

## Configuration Reference

### W2AClient Options

```typescript
const client = new W2AClient({
  channelName?: string;        // Default: "default"
  logLevel?: "silent" | "info" | "debug"; // Default: "info"
  signalBufferSize?: number;   // How many signals to buffer. Default: 100
  onError?: (err: Error) => void;
});
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `W2A_LOG_LEVEL` | Override log level (`silent`, `info`, `debug`) |
| `W2A_CHANNEL` | Default channel name |
| `WEATHER_API_KEY` | Example: API key for a weather sensor (sensor-specific) |

---

## Common Patterns

### Pattern: Agent reacts to high-priority signals only

```typescript
client.on("signal", (signal: W2ASignal) => {
  if (signal.priority === "high" || signal.priority === "critical") {
    triggerAgentAction(signal);
  }
});
```

### Pattern: Filter signals by sensor source

```typescript
client.on("signal", (signal: W2ASignal) => {
  if (signal.sensorId === "@world2agent/sensor-hackernews") {
    const { title, score } = signal.payload as { title: string; score: number };
    if (score > 200) {
      notifySlack(`Hot HN post: ${title}`);
    }
  }
});
```

### Pattern: Graceful shutdown

```typescript
process.on("SIGINT", async () => {
  console.log("Shutting down W2A client...");
  await client.stop();
  process.exit(0);
});
```

### Pattern: Logging all signals to a file

```typescript
import { appendFileSync } from "fs";

client.on("signal", (signal: W2ASignal) => {
  appendFileSync(
    "signals.ndjson",
    JSON.stringify(signal) + "\n",
    "utf-8"
  );
});
```

---

## SensorHub — Finding Sensors

Browse [world2agent.ai/hub](https://world2agent.ai/hub) for the full catalog. Categories include:

- 📰 **News** — Hacker News, frontier AI lab posts, RSS feeds
- 📈 **Markets** — stock prices, crypto, economic indicators
- 🚨 **Production** — error alerts, uptime, deployment events
- 🌦 **Weather** — real-time conditions, forecasts
- 🤖 **AI Labs** — OpenAI, Anthropic, Google announcements

Fallback search via npm:

```bash
npm search w2a-sensor
```

---

## Troubleshooting

**Signals not arriving**
- Confirm `client.start()` was called and awaited.
- Check sensor-specific config (API keys, filters).
- Set `logLevel: "debug"` to see internal sensor polling.

**Claude Code plugin not loading**
- Run `/reload-plugins` after install.
- Restart with `--dangerously-load-development-channels plugin:world2agent@world2agent-plugins`.

**TypeScript type errors on `payload`**
- `payload` is typed as `Record<string, unknown>` by design. Cast to your sensor's known shape: `signal.payload as { title: string; score: number }`.

**Sensor emitting too frequently**
- Increase `pollIntervalMs` in your sensor config.
- Add a deduplication check using `signal.id` or payload fingerprinting.

**Unknown signal types at runtime**
- Add a `default` case in your `switch` to log and handle gracefully.
- Check the sensor's docs on [SensorHub](https://world2agent.ai/hub) for all emitted `type` values.

---

## Key Links

- [SensorHub](https://world2agent.ai/hub) — browse all sensors
- [Signal format spec](./docs/signal-format.md)
- [Architecture deep dive](./docs/architecture.md)
- [Build a sensor guide](./docs/build-a-sensor.md)
- [Multi-sensor guide](./docs/multi-sensor.md)
- [Discord community](https://discord.gg/hDjaD8pX)
```
