```markdown
---
name: hackingtool-plugin-claude
description: Claude Code plugin wrapping 183+ pentesting & OSINT tools via ht_run.py with auto backend selection (native/WSL/Docker)
triggers:
  - recon a target domain
  - run pentesting tools
  - osint username lookup
  - scan for vulnerabilities
  - install hackingtool plugin
  - run nmap subfinder nuclei
  - enumerate subdomains and ports
  - find leaked secrets in git repo
---

# hackingtool — Claude Code Plugin

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

183 pentesting & OSINT tools available to Claude Code via a plugin wrapper around [Z4nzu/hackingtool](https://github.com/Z4nzu/hackingtool). Auto-selects backend: native Bash (Linux/macOS), WSL (Windows), or Docker images.

---

## Install

```bash
/plugin marketplace add AKCODEZ/hackingtool-plugin
/plugin install hackingtool@hackingtool-marketplace
```

Then just talk to Claude naturally:

```
"recon example.com"
"hunt the username johndoe across social platforms"
"scan my repo for leaked secrets"
"enumerate subdomains for target.com"
```

---

## Architecture

All tool invocations route through two core modules:

### `ht_env.py` — Backend detection

```python
# Backend priority: Docker > WSL > Native
import ht_env

env = ht_env.detect()
# Returns one of: "native", "wsl", "docker"
# env.run("nmap -sV 192.168.1.1") dispatches correctly
```

### `ht_run.py` — Tool executor

```python
import subprocess, json

result = subprocess.run(
    ["python3", "ht_run.py", "nmap", "--", "-sV", "192.168.1.1"],
    capture_output=True, text=True
)
output = json.loads(result.stdout)
# output = {"tool": "nmap", "backend": "docker", "stdout": "...", "stderr": "...", "returncode": 0}
```

CLI usage:

```bash
# Run a tool
python3 ht_run.py <tool_id> [-- <tool_args...>]

# Install a tool natively (Linux/WSL)
python3 ht_run.py nmap --install

# List available tool IDs
python3 ht_run.py --list
```

---

## Docker Image Map

The plugin auto-pulls purpose-built images. No manual `docker pull` needed.

| Tool ID | Docker Image |
|---|---|
| `nmap` | `instrumentisto/nmap` |
| `masscan` | `ilyaglow/masscan` |
| `rustscan` | `rustscan/rustscan` |
| `subfinder` | `projectdiscovery/subfinder` |
| `amass` | `caffix/amass` |
| `httpx` | `projectdiscovery/httpx` |
| `nuclei` | `projectdiscovery/nuclei` |
| `katana` | `projectdiscovery/katana` |
| `holehe` | `megadose/holehe` |
| `maigret` | `soxoj/maigret` |
| `spiderfoot` | `spiderfoot/spiderfoot` |
| `theharvester` | `secsi/theharvester` |
| `trufflehog` | `trufflesecurity/trufflehog` |
| `gitleaks` | `zricethezav/gitleaks` |
| `ffuf` | `secsi/ffuf` |
| `gobuster` | `devopsworks/gobuster` |
| `testssl` | `drwetter/testssl.sh` |
| `wafw00f` | `0xsauby/wafw00f` |
| `sqlmap` | `paoloo/sqlmap` |
| `impacket` | `rflathers/impacket` |
| `netexec` | `byt3bl33d3r/netexec` |
| `dnstwist` | `elceef/dnstwist` |
| *(fallback)* | `kalilinux/kali-rolling` |

---

## Common Usage Patterns

### Domain Recon Workflow

```python
import subprocess, json

def ht(tool_id, *args):
    cmd = ["python3", "ht_run.py", tool_id, "--"] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(r.stdout)

target = "example.com"

# 1. Passive subdomain enumeration
subs = ht("subfinder", "-d", target, "-silent")
print(subs["stdout"])

# 2. HTTP probe live hosts
httpx = ht("httpx", "-l", "-", "-silent")  # pipe subfinder output

# 3. Port scan
ports = ht("nmap", "-sV", "--open", "-T4", target)
print(ports["stdout"])

# 4. Vulnerability scan
vulns = ht("nuclei", "-u", f"https://{target}", "-severity", "critical,high")
print(vulns["stdout"])
```

### OSINT Username Hunt

```python
username = "johndoe"

# Check 3000+ sites
maigret = ht("maigret", username, "--html")
print(maigret["stdout"])

# Check email registration on 120+ sites
email = "johndoe@gmail.com"
holehe = ht("holehe", email)
print(holehe["stdout"])
```

### Secret Scanning

```python
import os

repo_path = "/path/to/repo"

# TruffleHog — verified secrets with entropy analysis
secrets = ht("trufflehog", "filesystem", repo_path, "--json")

# Gitleaks — fast pattern-based scan
leaks = ht("gitleaks", "detect", "--source", repo_path, "--report-format", "json")

for tool_result in [secrets, leaks]:
    if tool_result["returncode"] != 0:
        print(f"FINDINGS: {tool_result['stdout']}")
```

### SQL Injection Testing

```python
# Automated SQLi detection (only test targets you own/have permission)
target_url = "https://testphp.vulnweb.com/artists.php?artist=1"

result = ht("sqlmap", "-u", target_url, "--batch", "--level=2", "--risk=1")
print(result["stdout"])
```

### DNS Phishing Detection

```python
domain = "paypal.com"

# Find lookalike/typosquat domains
twist = ht("dnstwist", "--registered", domain, "--format", "json")
import json as _json
lookalikes = _json.loads(twist["stdout"])
for entry in lookalikes:
    print(f"{entry['fuzzer']}: {entry['domain']} → {entry.get('dns-a', 'no DNS')}")
```

---

## Backend-Aware Wrapper

```python
# ht_env.py usage — check what backend will be used
import subprocess

def get_backend():
    r = subprocess.run(
        ["python3", "ht_env.py", "--detect"],
        capture_output=True, text=True
    )
    return r.stdout.strip()  # "native" | "wsl" | "docker"

backend = get_backend()
print(f"Using backend: {backend}")

# For sudo-required tools on native/WSL:
# The plugin auto-retries with sudo on permission errors
# No manual sudo handling needed in calling code
```

---

## Tool Categories & IDs

| Category | Key Tool IDs |
|---|---|
| Information Gathering | `amass`, `subfinder`, `nmap`, `masscan`, `rustscan`, `httpx`, `spiderfoot`, `theharvester`, `maigret`, `holehe`, `gitleaks`, `trufflehog` |
| Web Attack | `ffuf`, `gobuster`, `sqlmap`, `nosqlmap`, `wafw00f`, `testssl`, `nuclei`, `katana` |
| OSINT | `maigret`, `holehe`, `infoga`, `redhawk`, `recondog`, `reconspider` |
| Wordlist/Passwords | `cupp`, `hashcat`, `john`, `haiti` |
| Wireless | `wifite`, `airgeddon`, `bettercap`, `wifiphisher` |
| Active Directory | `impacket`, `netexec` |
| Phishing Recon | `dnstwist` |
| Anonymity | `anonsurf`, `multitor` |

---

## Tool Flags Reference

| Flag | Meaning |
|---|---|
| `sudo` | Requires elevated privileges — auto-handled on native/WSL |
| `hw` | Requires physical hardware (WiFi adapter, Bluetooth) — Docker cannot satisfy |
| `interactive` | Needs a TTY / interactive session — Claude will note this |
| `long` | Long-running operation — may take minutes to hours |

🟢 = plug-and-play (56 tools) · 🟡 = environment-dependent (127 tools)

---

## Install Individual Tools (Native/WSL)

```bash
# Install a specific tool to the host system
python3 ht_run.py subfinder --install
python3 ht_run.py nuclei --install
python3 ht_run.py gitleaks --install

# Docker images need no install — pulled automatically on first use
```

---

## Troubleshooting

### Docker not found

```bash
# Verify Docker is running
docker info

# If not installed: https://docs.docker.com/get-docker/
# Plugin falls back to native/WSL if Docker unavailable
```

### Permission denied on native/WSL tools

```bash
# The plugin auto-retries with sudo
# If it still fails, check sudoers config:
sudo visudo
# Add: yourusername ALL=(ALL) NOPASSWD: /usr/bin/nmap
```

### WSL distro not detected

```bash
# List available WSL distros
wsl --list --verbose

# Plugin prefers Ubuntu/Kali — install if missing:
wsl --install -d Ubuntu
```

### Tool output is empty

```python
result = ht("subfinder", "-d", "example.com")
if result["returncode"] != 0:
    print("STDERR:", result["stderr"])
    # Check if Docker image pulled correctly:
    # docker run --rm projectdiscovery/subfinder -version
```

### Hardware-dependent tools in Docker

Tools with `hw` flag (wireless attacks, Bluetooth) **cannot run in Docker**. They require:
- Physical adapter passthrough to WSL2, or
- Running on bare-metal Linux

Claude will surface this constraint in the response when a `hw`-flagged tool is invoked.

---

## Environment Variables

```bash
# Optional: override Docker socket path
export DOCKER_HOST=unix:///var/run/docker.sock

# Optional: force a specific backend
export HT_BACKEND=docker   # "native" | "wsl" | "docker"

# Optional: WSL distro override
export HT_WSL_DISTRO=kali-linux

# Optional: extra docker run flags (e.g. network mode)
export HT_DOCKER_FLAGS="--network host"
```

---

## Responsible Use

- Only run these tools against systems you **own** or have **explicit written permission** to test.
- Many tools (nmap, sqlmap, masscan) trigger IDS/IPS alerts and may violate ToS or laws if used against unauthorized targets.
- The `--batch` flag on sqlmap and similar tools disables interactive prompts — use carefully.
- WiFi attack tools (`hw` flag) require compliance with local radio/wireless regulations.
```
