```markdown
---
name: sni-spoofing-dpi-bypass
description: Bypass Deep Packet Inspection (DPI) firewalls using IP/TCP header manipulation and SNI spoofing techniques in Python
triggers:
  - bypass DPI with SNI spoofing
  - how to use SNI spoofing to bypass firewall
  - TCP header manipulation to evade censorship
  - IP fragmentation DPI bypass Python
  - implement SNI spoofing script
  - circumvent deep packet inspection
  - spoof TLS SNI to bypass blocking
  - patterniha SNI spoofing setup
---

# SNI-Spoofing DPI Bypass

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

SNI-Spoofing is a Python-based tool that bypasses Deep Packet Inspection (DPI) systems by manipulating IP and TCP headers. It works by fragmenting TLS ClientHello packets so the SNI (Server Name Indication) field is split across multiple TCP segments, preventing DPI middleboxes from inspecting the destination hostname while still allowing the connection to succeed end-to-end.

---

## How It Works

DPI systems inspect the SNI field in TLS ClientHello to block connections to specific domains. SNI-Spoofing defeats this by:

1. **TCP segmentation** — Splitting the ClientHello across multiple TCP packets so the SNI is never in a single inspectable packet.
2. **IP fragmentation** — Fragmenting IP packets so DPI reassembly is bypassed.
3. **Raw socket manipulation** — Using Scapy or raw sockets to craft custom IP/TCP headers with precise control over fragmentation offsets and flags.

---

## Installation

### Requirements

- Python 3.8+
- Root/Administrator privileges (required for raw sockets)
- Linux (recommended; Windows has limited raw socket support)

```bash
git clone https://github.com/patterniha/SNI-Spoofing.git
cd SNI-Spoofing
pip install -r requirements.txt
```

### Dependencies

```bash
pip install scapy
```

On Linux, ensure `nftables` or `iptables` does not interfere with outgoing packets:

```bash
# Drop kernel RST packets for the port you're intercepting
sudo iptables -A OUTPUT -p tcp --tcp-flags RST RST -j DROP
```

---

## Project Structure

```
SNI-Spoofing/
├── main.py          # Entry point / core proxy logic
├── sni.py           # SNI extraction and packet splitting
├── utils.py         # Helper functions (checksum, headers)
└── requirements.txt
```

---

## Key Concepts & API

### SNI Extraction from TLS ClientHello

```python
def extract_sni(data: bytes) -> str | None:
    """
    Extract the SNI hostname from a raw TLS ClientHello payload.
    Returns None if not found.
    """
    try:
        if data[0] != 0x16:  # TLS Handshake
            return None
        if data[5] != 0x01:  # ClientHello
            return None

        # Skip TLS record header (5 bytes) + handshake header (4 bytes)
        # + client_version (2) + random (32) + session_id_length (1+n)
        pos = 9
        session_id_len = data[pos]
        pos += 1 + session_id_len

        # Skip cipher suites
        cipher_suites_len = int.from_bytes(data[pos:pos+2], 'big')
        pos += 2 + cipher_suites_len

        # Skip compression methods
        compression_len = data[pos]
        pos += 1 + compression_len

        # Extensions
        extensions_len = int.from_bytes(data[pos:pos+2], 'big')
        pos += 2
        end = pos + extensions_len

        while pos < end:
            ext_type = int.from_bytes(data[pos:pos+2], 'big')
            ext_len = int.from_bytes(data[pos+2:pos+4], 'big')
            pos += 4
            if ext_type == 0x0000:  # SNI extension
                # server_name_list_length (2) + name_type (1) + name_length (2)
                name_len = int.from_bytes(data[pos+3:pos+5], 'big')
                return data[pos+5:pos+5+name_len].decode('utf-8')
            pos += ext_len
    except Exception:
        return None
    return None
```

### TCP Segmentation of ClientHello

```python
from scapy.all import IP, TCP, Raw, send

def send_fragmented_client_hello(
    dst_ip: str,
    dst_port: int,
    src_port: int,
    seq: int,
    client_hello: bytes,
    split_position: int = None
):
    """
    Split TLS ClientHello into two TCP segments at split_position.
    Default split: right before the SNI hostname bytes.
    """
    if split_position is None:
        # Split in the middle of the packet to bisect SNI
        split_position = len(client_hello) // 2

    part1 = client_hello[:split_position]
    part2 = client_hello[split_position:]

    pkt1 = (
        IP(dst=dst_ip) /
        TCP(dport=dst_port, sport=src_port, seq=seq, flags="PA") /
        Raw(load=part1)
    )
    pkt2 = (
        IP(dst=dst_ip) /
        TCP(dport=dst_port, sport=src_port, seq=seq + len(part1), flags="PA") /
        Raw(load=part2)
    )

    send(pkt1, verbose=False)
    send(pkt2, verbose=False)
```

### IP Fragmentation Approach

```python
from scapy.all import IP, TCP, Raw, fragment, send

def send_ip_fragmented(dst_ip: str, dst_port: int, payload: bytes):
    """
    Send payload as IP fragments to bypass DPI reassembly.
    Fragment size of 8 bytes ensures SNI is split across fragments.
    """
    packet = IP(dst=dst_ip) / TCP(dport=dst_port) / Raw(load=payload)
    fragments = fragment(packet, fragsize=8)
    for frag in fragments:
        send(frag, verbose=False)
```

---

## Running the Tool

Most usage requires root:

```bash
sudo python3 main.py --host <TARGET_DOMAIN> --port 443
```

### Common CLI Options

```bash
# Basic usage - connect through SNI-spoofed tunnel
sudo python3 main.py --host example.com --port 443

# Specify local proxy port to bind
sudo python3 main.py --host example.com --port 443 --local-port 8080

# Use IP fragmentation mode
sudo python3 main.py --host example.com --port 443 --mode fragment

# Use TCP split mode (default)
sudo python3 main.py --host example.com --port 443 --mode split

# Specify custom split offset (byte position to split ClientHello)
sudo python3 main.py --host example.com --port 443 --split-at 40
```

---

## Full Working Example: Transparent Proxy with SNI Splitting

```python
#!/usr/bin/env python3
"""
Minimal transparent proxy that intercepts TLS ClientHello
and resends it split across two TCP segments.
Requires: pip install scapy
Run as root.
"""

import socket
import threading
from scapy.all import IP, TCP, Raw, send, sniff, conf

conf.verb = 0  # Suppress Scapy output

TARGET_HOST = "example.com"
TARGET_PORT = 443
LOCAL_PORT = 8080


def extract_sni(data: bytes) -> str | None:
    try:
        if len(data) < 6 or data[0] != 0x16 or data[5] != 0x01:
            return None
        pos = 9
        pos += 1 + data[pos]
        cipher_len = int.from_bytes(data[pos:pos+2], 'big')
        pos += 2 + cipher_len
        pos += 1 + data[pos]
        pos += 2  # skip extensions length field
        while pos < len(data) - 4:
            ext_type = int.from_bytes(data[pos:pos+2], 'big')
            ext_len = int.from_bytes(data[pos+2:pos+4], 'big')
            pos += 4
            if ext_type == 0:
                name_len = int.from_bytes(data[pos+3:pos+5], 'big')
                return data[pos+5:pos+5+name_len].decode()
            pos += ext_len
    except Exception:
        pass
    return None


def handle_client(client_sock: socket.socket, target_ip: str):
    try:
        client_hello = client_sock.recv(4096)
        sni = extract_sni(client_hello)
        print(f"[*] Intercepted ClientHello, SNI: {sni}")

        # Connect to real server with raw socket for first packet
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.connect((target_ip, TARGET_PORT))

        split_at = len(client_hello) // 2
        raw.send(client_hello[:split_at])
        raw.send(client_hello[split_at:])

        # Relay remaining data bidirectionally
        def relay(src, dst):
            try:
                while chunk := src.recv(4096):
                    dst.send(chunk)
            except Exception:
                pass

        t1 = threading.Thread(target=relay, args=(raw, client_sock), daemon=True)
        t2 = threading.Thread(target=relay, args=(client_sock, raw), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    finally:
        client_sock.close()


def main():
    target_ip = socket.gethostbyname(TARGET_HOST)
    print(f"[*] Resolving {TARGET_HOST} -> {target_ip}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", LOCAL_PORT))
    server.listen(10)
    print(f"[*] Listening on 127.0.0.1:{LOCAL_PORT}")

    while True:
        client, addr = server.accept()
        print(f"[+] Connection from {addr}")
        threading.Thread(
            target=handle_client,
            args=(client, target_ip),
            daemon=True
        ).start()


if __name__ == "__main__":
    main()
```

---

## iptables Setup (Required for Raw Socket Mode)

When using Scapy to send raw packets, the kernel may send RST packets in parallel. Suppress them:

```bash
# Prevent kernel from sending RST on the target port
sudo iptables -I OUTPUT -p tcp --tcp-flags RST RST --dport 443 -j DROP

# Restore after use
sudo iptables -D OUTPUT -p tcp --tcp-flags RST RST --dport 443 -j DROP
```

---

## Configuration Patterns

### Environment-based Configuration

```python
import os

config = {
    "target_host": os.environ.get("SNI_TARGET_HOST", "example.com"),
    "target_port": int(os.environ.get("SNI_TARGET_PORT", "443")),
    "local_port":  int(os.environ.get("SNI_LOCAL_PORT", "8080")),
    "split_mode":  os.environ.get("SNI_SPLIT_MODE", "tcp"),   # tcp | ip_frag
    "split_offset": int(os.environ.get("SNI_SPLIT_OFFSET", "0")),  # 0 = auto
}
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Permission denied` on socket | Raw sockets need root | Run with `sudo` |
| Connection RST immediately | Kernel sends RST alongside Scapy | Add `iptables` RST drop rule |
| SNI not found / returns `None` | Packet captured before TLS layer | Ensure you're reading post-TCP-handshake data |
| DPI still blocking | Split point lands outside SNI field | Use `--split-at` to target SNI bytes precisely |
| Works locally but not remotely | ISP-level DPI uses IP reassembly | Switch to `--mode fragment` (IP fragmentation) |
| `ImportError: scapy` | Scapy not installed | `pip install scapy` |

---

## Platform Notes

- **Linux**: Full support for raw sockets and IP fragmentation. Recommended platform.
- **macOS**: Raw sockets work but may require disabling SIP for some operations.
- **Windows**: Raw socket support is severely limited; TCP split mode may work via WinDivert (`pydivert`).

---

## Legal & Ethical Notice

This tool is intended for research, circumventing censorship in restrictive regions, and legitimate privacy use cases. Misuse against systems you do not own or have permission to test is illegal. Always comply with local laws.
```
