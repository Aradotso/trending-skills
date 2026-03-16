# Trending Skills

Auto-generated agent skills from GitHub's trending open source projects, by [ara.so](https://ara.so).

## Install all skills

```bash
npx skills add Aradotso/trending-skills
```

## Install a specific skill

```bash
npx skills add Aradotso/trending-skills --skill lightpanda-browser
```

## Skills Index

| Skill | skills.sh | Source | Date |
|-------|-----------|--------|------|
| [lightpanda-browser](skills/lightpanda-browser/) | [install](https://skills.sh/Aradotso/trending-skills/lightpanda-browser) | [lightpanda-io/browser](https://github.com/lightpanda-io/browser) | 2026-03-15 |
| [gstack-workflow-assistant](skills/gstack-workflow-assistant/) | [install](https://skills.sh/Aradotso/trending-skills/gstack-workflow-assistant) | [garrytan/gstack](https://github.com/garrytan/gstack) | 2026-03-15 |
| [openclaw-config](skills/openclaw-config/) | [install](https://skills.sh/Aradotso/trending-skills/openclaw-config) | — | — |
| [openclaw-control-center](skills/openclaw-control-center/) | [install](https://skills.sh/Aradotso/trending-skills/openclaw-control-center) | [TianyiDataScience/openclaw-control-center](https://github.com/TianyiDataScience/openclaw-control-center) | 2026-03-15 |
| [pi-autoresearch-loop](skills/pi-autoresearch-loop/) | [install](https://skills.sh/Aradotso/trending-skills/pi-autoresearch-loop) | [davebcn87/pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) | 2026-03-15 |
| [chrome-cdp-live-browser](skills/chrome-cdp-live-browser/) | [install](https://skills.sh/Aradotso/trending-skills/chrome-cdp-live-browser) | [pasky/chrome-cdp-skill](https://github.com/pasky/chrome-cdp-skill) | 2026-03-15 |
| [gsd-2-agent-framework](skills/gsd-2-agent-framework/) | [install](https://skills.sh/Aradotso/trending-skills/gsd-2-agent-framework) | [gsd-build/gsd-2](https://github.com/gsd-build/gsd-2) | 2026-03-15 |
| [metaclaw-evolving-agent](skills/metaclaw-evolving-agent/) | [install](https://skills.sh/Aradotso/trending-skills/metaclaw-evolving-agent) | [aiming-lab/MetaClaw](https://github.com/aiming-lab/MetaClaw) | 2026-03-15 |
| [aris-autonomous-ml-research](skills/aris-autonomous-ml-research/) | [install](https://skills.sh/Aradotso/trending-skills/aris-autonomous-ml-research) | [wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) | 2026-03-16 |
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
