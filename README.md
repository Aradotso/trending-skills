# Trending Skills

Auto-generated agent skills from GitHub's trending open source projects, by [ara.so](https://ara.so).

Every 15 minutes, a GitHub Actions workflow checks what's trending on GitHub and creates a comprehensive agent skill for the top project. Skills are published to [skills.sh](https://skills.sh/Aradotso/trending-skills) and installable by any AI coding agent.

## Install all skills

```bash
npx skills add Aradotso/trending-skills
```

## Install a specific skill

```bash
npx skills add Aradotso/trending-skills --skill lightpanda-browser
```

## Skills Index

| Skill | Description | Source | Date |
|-------|-------------|--------|------|
| [lightpanda-browser](skills/lightpanda-browser/) | Headless browser built in Zig for AI and automation | [lightpanda-io/browser](https://github.com/lightpanda-io/browser) | 2026-03-15 |
| [gstack-workflow-assistant](skills/gstack-workflow-assistant/) | Claude Code workflow tools by Garry Tan | [garrytan/gstack](https://github.com/garrytan/gstack) | 2026-03-15 |
| [openclaw-config](skills/openclaw-config/) | Manage OpenClaw bot configuration — channels, agents, security, autopilot | — | — |
| [openclaw-control-center](skills/openclaw-control-center/) | Turn OpenClaw into a local control center | [TianyiDataScience/openclaw-control-center](https://github.com/TianyiDataScience/openclaw-control-center) | 2026-03-15 |
| [pi-autoresearch-loop](skills/pi-autoresearch-loop/) | Autonomous experiment loop extension | [davebcn87/pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) | 2026-03-15 |
| [chrome-cdp-skill](skills/chrome-cdp-skill/) | Give your AI agent access to your live Chrome session — works out of the box, connects to tabs you already have open | [pasky/chrome-cdp-skill](https://github.com/pasky/chrome-cdp-skill) | 2026-03-15 |
| [gsd-2](skills/gsd-2/) | A powerful meta-prompting, context engineering and spec-driven development system that enables agents to work for long periods of time autonomously without losing track of the big picture | [gsd-build/gsd-2](https://github.com/gsd-build/gsd-2) | 2026-03-15 |
<!-- SKILL_INDEX -->

## How it works

1. GitHub Actions cron runs every 15 minutes
2. Fetches the hottest new repos (created in the last 7 days, sorted by stars)
3. Skips repos that already have a skill
4. Uses Claude Sonnet 4.6 to generate a comprehensive SKILL.md from the repo's README
5. Commits, pushes, and registers on skills.sh

## By ara.so

[Ara](https://ara.so) — instant AI agent environments in the cloud.

## License

MIT
