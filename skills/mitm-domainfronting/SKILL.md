```markdown
---
name: mitm-domainfronting
description: MITM proxy that receives unencrypted traffic locally then forwards it via Domain Fronting using Xray-core, enabling access to restricted services without a traditional proxy server.
triggers:
  - set up MITM domain fronting
  - configure xray domain fronting
  - bypass censorship with domain fronting
  - MITM proxy with self-signed certificate
  - xray MITM configuration
  - domain fronting without server
  - access google services with domain fronting
  - set up v2rayN domain fronting
---

# MITM-DomainFronting

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What This Project Does

MITM-DomainFronting is a technique (not a traditional proxy server) that:
1. **Intercepts** unencrypted HTTPS traffic locally using a Man-in-the-Middle (MITM) setup with a self-signed certificate
2. **Re-sends** that traffic to the real destination using a fake/fronted SNI (Server Name Indication), bypassing censorship/firewalls

This works **without a remote proxy server** — it uses Xray-core's built-in MITM and Domain Fronting features. Currently enables access to Google services (Meet, Drive, etc.) from restricted regions.

**What it cannot do:**
- Fully replace a VPN or proxy for general internet access
- Access YouTube videos (separate service infrastructure)
- Access Gemini (Iran IPs are sanctioned)

---

## Architecture Overview

```
Browser (HTTPS) → [MITM: fake cert intercepts, decrypts] → [Domain Fronting: re-encrypts with fronted SNI] → Google Servers
```

- **MITM layer**: Xray acts as a TLS terminator using your self-signed certificate
- **Domain Fronting layer**: Xray re-establishes TLS to the real server using a CDN-friendly SNI

---

## Prerequisites

- **Windows**: v2rayN + Xray-core
- **Android**: v2rayNG with HEV TUN enabled
- **Linux/macOS**: Xray-core directly
- A self-signed certificate (`.crt` + `.key`)

---

## Step 1: Generate a Self-Signed Certificate (Windows)

Use the provided `certificate-generator.bat` in the `v2rayN-windows-64\bin` folder:

```bat
@echo off
REM certificate-generator.bat
REM Place this in v2rayN-windows-64\bin and run it there

openssl req -x509 -newkey rsa:4096 -keyout mycert.key -out mycert.crt -sha256 -days 3650 -nodes -subj "/C=US/ST=State/L=City/O=Org/CN=localhost"

echo Done! mycert.crt and mycert.key created.
pause
```

> ⚠️ **Security Warning**: Never share your `mycert.key` (private key) with anyone. Never use someone else's `mycert.crt`. Always generate your own.

**Alternative (online generator for Android)**:
Use https://regery.com/en/security/ssl-tools/self-signed-certificate-generator — download both files and rename them to `mycert.crt` and `mycert.key`.

---

## Step 2: Install the Certificate as Trusted Root

### Windows (System-wide)
1. Right-click `mycert.crt` → **Install Certificate**
2. Select **Local Machine**
3. Choose **Place all certificates in the following store**
4. Select **Trusted Root Certification Authorities**
5. Confirm

### Windows (Chrome only)
```
Settings → Privacy and security → Security → Manage certificates →
Manage imported certificates from Windows → Trusted Root Certification Authorities →
Import → Select mycert.crt → Place all certificates in the following store →
Select "Trusted Root Certification Authorities"
```

### Android (Non-root)
```
Settings → Security and privacy → More security settings →
Install from device storage → CA Certificate → Install anyway →
Select mycert.crt
```

Verify installation:
```
Settings → Security and privacy → More security settings →
View security certificates → User
```

### Android Firefox (additional step)
```
firefox → Settings → About Firefox → Tap logo 5 times →
Settings → Secret Settings → Toggle "Use third party CA certificates"
```

---

## Step 3: Core Configuration File

The main config file `MITM-DomainFronting.json` for Xray-core:

```json
{
  "log": {
    "loglevel": "warning"
  },
  "inbounds": [
    {
      "tag": "http-in",
      "port": 10809,
      "protocol": "http",
      "settings": {
        "allowTransparent": true
      },
      "sniffing": {
        "enabled": true,
        "destOverride": ["http", "tls"]
      }
    }
  ],
  "outbounds": [
    {
      "tag": "mitm-out",
      "protocol": "freedom",
      "settings": {}
    }
  ],
  "policy": {
    "levels": {
      "0": {
        "handshakeMitm": true
      }
    }
  },
  "mitm": {
    "enabled": true,
    "listenPort": 10810,
    "certFile": "mycert.crt",
    "keyFile": "mycert.key",
    "domainFronting": {
      "enabled": true,
      "frontDomain": "www.googleapis.com",
      "targetDomain": "www.google.com"
    }
  }
}
```

> **Note**: File paths for `certFile` and `keyFile` are relative to the Xray binary location (i.e., `v2rayN-windows-64\bin\`).

---

## Step 4: v2rayN Setup (Windows)

1. Download latest [v2rayN](https://github.com/2dust/v2rayN/releases) (`v2rayN-windows-64.zip`)
2. Extract it
3. Copy `certificate-generator.bat`, `mycert.crt`, `mycert.key`, and `MITM-DomainFronting.json` into `v2rayN-windows-64\bin\`
4. Open v2rayN → **Configuration** → **Add a custom configuration**
5. Set:
   - **Name**: anything descriptive
   - **Config file**: `MITM-DomainFronting.json`
   - **Core type**: `xray`
   - **Socks port**: leave empty
6. Select the config → **Set system proxy**

---

## Step 5: v2rayNG Setup (Android)

1. Install latest [v2rayNG](https://github.com/2dust/v2rayNG/releases)
2. Go to **Asset files** → Import both `mycert.crt` and `mycert.key`
3. Import `MITM-DomainFronting.json` via **Import from locally**
4. Enable **HEV TUN FEATURE** in v2rayNG settings
5. Connect

> ⚠️ On non-rooted Android, only browser-based access works. Standalone apps (Google Meet app, Drive app) won't use this tunnel — use the browser versions instead.

---

## Running Xray Directly (Linux/macOS)

```bash
# Place mycert.crt, mycert.key, and MITM-DomainFronting.json in the same directory as xray binary

# Download xray
wget https://github.com/XTLS/Xray-core/releases/latest/download/Xray-linux-64.zip
unzip Xray-linux-64.zip -d xray-bin
cd xray-bin

# Copy your cert files here
cp /path/to/mycert.crt .
cp /path/to/mycert.key .
cp /path/to/MITM-DomainFronting.json .

# Run
./xray run -config MITM-DomainFronting.json
```

Set system proxy to `http://127.0.0.1:10809` in your OS network settings.

---

## Common Patterns

### Verify the certificate is trusted
```bash
# Linux/macOS
openssl verify -CAfile mycert.crt mycert.crt
# Should output: mycert.crt: OK
```

### Check Xray is running and listening
```bash
# Check port is open
netstat -an | grep 10809
# or
ss -tlnp | grep 10809
```

### Test connectivity through the proxy
```bash
# Set proxy and test Google
curl -x http://127.0.0.1:10809 https://www.google.com -v
```

### Generate cert with specific domain SANs (advanced)
```bat
REM certificate-generator-advanced.bat
openssl req -x509 -newkey rsa:4096 ^
  -keyout mycert.key ^
  -out mycert.crt ^
  -sha256 -days 3650 -nodes ^
  -subj "/CN=*.google.com" ^
  -addext "subjectAltName=DNS:*.google.com,DNS:*.googleapis.com,DNS:*.gstatic.com"
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Browser shows certificate error | Certificate not installed as trusted root | Re-do Step 2; verify in cert manager |
| Google Meet/Drive won't load | Wrong fronting domain | Check `frontDomain` in config matches a CDN endpoint |
| Android apps don't work | Non-root limitation | Use browser instead of native apps |
| `mycert.crt not found` error | Wrong working directory | Ensure cert files are in same folder as xray binary |
| Connection refused on port 10809 | Xray not running | Check xray process; check for port conflicts |
| YouTube videos don't work | Separate CDN infrastructure | Not supported by this method |
| Gemini doesn't work | IP sanctions | Not supported from Iran IPs |

### Enable debug logging
```json
{
  "log": {
    "loglevel": "debug",
    "access": "access.log",
    "error": "error.log"
  }
}
```

---

## How Domain Fronting Works (Technical)

```
1. Browser sends:  TLS ClientHello → SNI: "www.google.com"
                   HTTP Host: "www.google.com"

2. MITM intercepts: Xray presents mycert.crt, terminates TLS
                    Reads plaintext HTTP request

3. Domain Fronting: Xray opens new TLS to CDN
                    SNI in TLS: "www.googleapis.com"  ← firewall sees this (allowed)
                    Host header: "www.google.com"      ← CDN routes to this (real target)
```

The firewall/DPI only sees the outer SNI (`googleapis.com`), which is typically not blocked. The CDN then routes based on the HTTP Host header to the real destination.

---

## Project References

- Original Python implementation: https://github.com/patterniha/MMDF
- Xray-core integration PR: https://github.com/XTLS/Xray-core/issues/4348
- v2rayN releases: https://github.com/2dust/v2rayN/releases
- v2rayNG releases: https://github.com/2dust/v2rayNG/releases
```
