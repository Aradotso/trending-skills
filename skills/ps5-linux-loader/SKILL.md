```markdown
---
name: ps5-linux-loader
description: Linux payload for PS5 consoles implementing HV exploit and custom bootloader to run full Linux on PS5 Phat hardware
triggers:
  - "run linux on ps5"
  - "ps5 linux loader"
  - "boot linux on playstation 5"
  - "ps5 jailbreak linux"
  - "compile ps5 linux payload"
  - "ps5 hv exploit linux"
  - "install linux ps5 usb"
  - "ps5 linux bootloader"
---

# PS5 Linux Loader

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## What It Does

`ps5-linux-loader` is a C-based payload that exploits a patched HV (Hypervisor) vulnerability on PS5 Phat consoles running firmware 3.xx or 4.xx to boot a full Linux environment. It implements:

- A custom bootloader that takes over from the PS5 OS
- HV exploit chain to escape the GameOS VM and run bare-metal
- Boot configuration via `cmdline.txt` and `vram.txt` on a FAT32 USB partition
- Support for USB boot drives, M.2 SSD installation, GPU/CPU boost, and fan control

**Supported hardware:** PS5 Phat only  
**Supported firmwares:** 3.00, 3.10, 3.20, 3.21, 4.00, 4.02, 4.03, 4.50, 4.51

---

## Prerequisites

### Build Dependencies

```bash
# Install PS5 Payload SDK (required)
# See: https://github.com/ps5-payload-dev/sdk

# On Debian/Ubuntu (x86-64 host):
sudo apt install build-essential git

# For ARM64 Linux hosts, install cross-compilation tools:
sudo apt install gcc-x86-64-linux-gnu binutils-x86-64-linux-gnu
```

### Required Hardware

- PS5 Phat console on firmware 3.xx or 4.xx
- USB drive ≥ 64GB (external SSD recommended)
- USB Ethernet or WLAN adapter
- USB keyboard and mouse
- A PC on the same network as the PS5

### Optional Hardware

- M.2 SSD (PS5-compatible) for dedicated Linux partition
- Bluetooth dongle for DualSense controller
- Capture card for external display testing

---

## Building the Payload

```bash
# Clone the repository
git clone https://github.com/ps5-linux/ps5-linux-loader
cd ps5-linux-loader

# Build (requires ps5-payload-sdk installed)
make
# Output: ps5-linux-loader.elf
```

Alternatively, download a prebuilt `ps5-linux-loader.elf` from the [releases page](https://github.com/ps5-linux/ps5-linux-loader/releases/).

---

## Building the Linux Image

```bash
# Clone the image builder
git clone https://github.com/ps5-linux/ps5-linux-image
cd ps5-linux-image
chmod +x ./build_image.sh

# Build Ubuntu 26.04 image (default)
./build_image.sh --distro ubuntu2604

# Output image location:
# output/ps5-ubuntu2604.img
```

**Windows (WSL2):**
```bash
# Install WSL2 first (PowerShell as admin):
wsl --install

# In WSL:
sudo apt update && sudo apt install docker.io -y
sudo service docker start
sudo usermod -aG docker $USER

# Restart WSL, then:
cd ~/
git clone https://github.com/ps5-linux/ps5-linux-image
cd ps5-linux-image
chmod +x ./build_image.sh
./build_image.sh --distro ubuntu2604
```

---

## Flashing the Image to USB

**Linux/macOS:**
```bash
# Identify your drive
lsblk           # Linux
diskutil list   # macOS

# Flash (replace sdX with your drive identifier)
sudo dd if=output/ps5-ubuntu2604.img of=/dev/sdX bs=4M status=progress conv=fsync
```

**Windows (WSL2 + usbipd):**
```powershell
# In PowerShell as admin:
winget install usbipd
usbipd list
usbipd bind --busid 5-3        # replace 5-3 with your busid
usbipd attach --busid 5-3 --wsl
```

```bash
# In WSL:
lsblk
sudo wipefs -a /dev/sdX
sudo dd if=output/ps5-ubuntu2604.img of=/dev/sdX bs=4M status=progress
```

---

## PS5 Configuration (Required)

Before running the exploit, configure the PS5:

1. **Enable USB power in Rest Mode:**  
   `Settings → System → Power Saving → Features Available in Rest Mode → Supply Power to USB Ports → Always`

2. **Disable HDMI Device Link:**  
   `Settings → HDMI → Enable HDMI Device Link` (turn off)

3. **Recommended — Disable auto-updates:**  
   `Settings → System Software → System Software Update and Settings`

---

## Running the Exploit

### Step 1: Set up Fake DNS + HTTPS Server

```bash
# Clone the exploit host
git clone https://github.com/idlesauce/umtx2
cd umtx2

# Edit dns.conf to point manuals.playstation.net to your PC's IP
# Then run:
python fakedns.py -c dns.conf    # Terminal 1
python host.py                    # Terminal 2
```

### Step 2: Point PS5 DNS to your PC

On PS5: `Settings → Network → Advanced Settings → Primary DNS` → set to your PC's IP  
Leave Secondary DNS as `0.0.0.0`

### Step 3: Trigger the exploit

On PS5: Go to user manual in Settings, accept the untrusted certificate prompt, and run.

### Step 4: Send the payload

```bash
# Find PS5 IP at: Settings → Network → View Connection Status
# Replace 192.168.178.127 with your PS5's IP

socat -t 99999999 - TCP:192.168.178.127:9021 < ps5-linux-loader.elf
```

### Step 5: Boot into Linux

1. PS5 will automatically enter rest mode after payload is sent
2. Wait for the **orange LED to stop blinking** and become **static**
3. Press the power button once
4. **White LED = successful Linux boot**

---

## Boot Configuration

The FAT32 partition on the boot USB contains two config files:

### `cmdline.txt` — Kernel command line

```bash
# Default content example:
root=LABEL=ubuntu2604 rw quiet splash

# Force 1080p output (useful for display compatibility issues):
amdgpu.force_1080p=1

# To boot from M.2 SSD instead of USB:
root=LABEL=ubuntu2604-m2
```

### `vram.txt` — VRAM allocation

```
# Default: 512MB (enables Dynamic VRAM allocation)
0x20000000

# Increase VRAM to 1GB:
0x40000000
```

---

## M.2 SSD Installation

```bash
# Step 1: Attach M.2 SSD and format via PS5 OS

# Step 2: Boot Linux, then initialize M.2
sudo apt install zlib1g-dev
git clone https://github.com/ps5-linux/ps5-linux-tools
cd ps5-linux-tools
gcc -o m2_init m2_init.c -lz
sudo ./m2_init

# Step 3: Reboot
sudo reboot

# Step 4: After relaunching Linux, install image to M.2
chmod +x ./m2_install.sh
sudo ./m2_install.sh --install /path/to/ps5-ubuntu2604.img

# Step 5: Boot from M.2
chmod +x ./m2_exec.sh
sudo ./m2_exec.sh
```

To make M.2 the default boot target, edit `/boot/efi/cmdline.txt`:
```
root=LABEL=ubuntu2604-m2
```

---

## Fan & CPU/GPU Boost Control

```bash
cd ps5-linux-tools

# Compile the control tool
gcc -o ps5_control ps5_control.c

# Enable fan (always enable before boost)
sudo ./ps5_control --fan on

# Enable CPU (3500 MHz) and GPU (2230 MHz) boost
sudo ./ps5_control --boost on

# Disable boost
sudo ./ps5_control --boost off

# Disable fan
sudo ./ps5_control --fan off
```

> ⚠️ Always enable the fan before enabling boost, matching official PS5 OS behavior.

---

## First Boot Setup

```bash
# 1. Disable screen saver (currently buggy — do this first)

# 2. Re-enable network connection if no internet
# Toggle off/on your Wired or WLAN connection in network settings

# 3. Install Firefox
sudo snap install firefox
sudo snap refresh mesa-2404 --channel=latest/edge

# 4. Clone ps5-linux-tools for fan/boost control
git clone https://github.com/ps5-linux/ps5-linux-tools
```

---

## Troubleshooting

### White LED but black screen

```bash
# Try setting 1080p forced mode in cmdline.txt:
amdgpu.force_1080p=1
```

- Try different monitors or cables
- Try a capture card
- Test with different resolution monitors
- Check Discord for EDID-specific help

### PS5 boots back into PS5 OS instead of Linux

- You pressed the power button **before** the orange LED became static
- Re-run the exploit and wait longer after rest mode LED becomes static

### Build fails (ARM64 host)

```bash
sudo apt install gcc-x86-64-linux-gnu binutils-x86-64-linux-gnu
make
```

### M.2 asks to reformat after m2_init

- Report your M.2 model and storage size to the Discord server
- Known issue with certain M.2 models

### No internet in Linux

- Re-toggle your network adapter (disable, then enable) in network settings
- USB Ethernet or WLAN adapter required — built-in PS5 NIC/WLAN not yet supported

### socat command not found

```bash
# Install socat on Linux/macOS:
sudo apt install socat        # Debian/Ubuntu
brew install socat            # macOS
```

---

## Firmware-Specific Notes

| Firmware | Support | M.2 Support |
|----------|---------|-------------|
| 3.00, 3.10, 3.20, 3.21 | ✅ | ❌ |
| 4.00, 4.02, 4.03, 4.50, 4.51 | ✅ | ✅ |
| 1.xx, 2.xx | Planned | Unknown |
| 5.xx | Planned (VM only, limited) | Unknown |
| 6.xx+ | ❌ Not supported | ❌ |

**To downgrade/target a specific firmware:**  
Download PUP from [darthsternie.net/ps5-firmwares](https://darthsternie.net/ps5-firmwares/) and follow the [official Sony reinstall guide](https://www.playstation.com/en-us/support/hardware/reinstall-playstation-system-software-safe-mode).

---

## Hardware Specs Available Under Linux

| Component | Spec |
|-----------|------|
| CPU | 8 cores / 16 threads @ 3.5 GHz |
| GPU | AMD @ 2.23 GHz |
| Video Output | HDMI 4K @ 60Hz |
| Resolutions | 1080p, 1440p, 2160p @ 60Hz |
| USB Ports | All ports (front bottom Type-C, rear Type-A; front top Type-A is USB 2.0) |
| Storage | USB drive or M.2 SSD |
| Controllers | DualSense via Bluetooth dongle |

---

## Known Limitations / Bugs

- Screen saver is buggy — disable it after first boot
- HDMI audio does not work on some monitors
- HDMI 1440p and 2160p video issues on some monitors
- Built-in PS5 Bluetooth not supported (use a dongle)
- Built-in PS5 NIC/WLAN not supported (use USB adapter)
- No standby/resume support
- No dual-boot — must re-run exploit each time to boot Linux
- Internal SSD is not modified by this process

---

## Community & Support

- [Discord server](https://discord.gg/PeMGVB7BAm) — help, tips, development, issue reporting
- [AMD BC250 Documentation](https://elektricm.github.io/amd-bc250-docs/) — many tips apply to PS5
- [ps5-linux-tools](https://github.com/ps5-linux/ps5-linux-tools) — fan, boost, M.2 tools
- [ps5-linux-image](https://github.com/ps5-linux/ps5-linux-image) — Linux image builder
```
