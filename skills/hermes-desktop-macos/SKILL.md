```markdown
---
name: hermes-desktop-macos
description: Native macOS SSH workspace for Hermes Agent — sessions, terminal, files, usage, and skills over direct SSH
triggers:
  - set up Hermes Desktop on my Mac
  - connect Hermes Desktop to my SSH host
  - configure Hermes Desktop SSH connection
  - browse Hermes sessions in the desktop app
  - edit Hermes memory files from Mac
  - build Hermes Desktop from source
  - troubleshoot Hermes Desktop SSH connection
  - add a new connection profile in Hermes Desktop
---

# Hermes Desktop macOS

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

Hermes Desktop is a native macOS app (Swift/SwiftUI, macOS 14+) that gives Hermes Agent users a first-class Mac workspace over direct SSH. It reads sessions from `~/.hermes/state.db`, edits canonical memory files (`USER.md`, `MEMORY.md`, `SOUL.md`), browses skills, shows token usage dashboards, and embeds a real SSH terminal — all without a gateway API, local file mirrors, or a remote helper service.

---

## Installation

### Option A: Download the release binary

```bash
# 1. Download from GitHub Releases
# 2. Unzip
unzip HermesDesktop.app.zip

# 3. Move to Applications
mv HermesDesktop.app /Applications/

# 4. Verify the bundle signature
codesign --verify --deep --strict /Applications/HermesDesktop.app

# 5. Check SHA-256 matches the release asset
shasum -a 256 HermesDesktop.app.zip
```

> **First-launch gatekeeper bypass** (app is not yet notarized):
> Right-click → Open, or go to **System Settings → Privacy & Security → Open Anyway**.

### Option B: Build from source

```bash
git clone https://github.com/dodo-reach/hermes-desktop
cd hermes-desktop

# Requires Xcode 15+ and macOS 14 SDK
./scripts/build-macos-app.sh

# The built app lands in the repo root or DerivedData — check script output
open HermesDesktop.app
```

---

## Requirements

| Requirement | Detail |
|---|---|
| macOS | 14 Sonoma or newer |
| SSH auth | Key-based, no interactive prompts from this Mac |
| Host key | Already accepted once in Terminal |
| Remote | `python3` available in the SSH environment |
| Hermes data | `~/.hermes/` present on the remote host |

Quick sanity check — if this works without prompts, the app will work:

```bash
ssh your-host echo ok
# → ok
```

---

## Connecting to a Hermes Host

### Using an SSH alias (recommended)

Add an entry to `~/.ssh/config` on your Mac:

```sshconfig
Host hermes-pi
  HostName 192.168.1.42
  User pi
  IdentityFile ~/.ssh/id_ed25519

Host hermes-vps
  HostName vps.example.com
  User deploy
  Port 2222
  IdentityFile ~/.ssh/id_ed25519

Host hermes-local
  HostName localhost
  User alex
```

In the app: **Connections → New Profile → SSH alias → `hermes-pi`**. Leave Host/User/Port blank.

### Using explicit host details

In the app:
- **Host or IP**: `vps.example.com`
- **User**: `deploy`
- **Port**: `2222`

### Testing the connection

Click **Test** before **Use Host**. The preflight check verifies:
1. SSH target is reachable
2. Authentication completes without prompts
3. `python3` exists in the remote SSH environment

---

## App Sections

### Overview

Displays on first connect:
- Remote `$HOME` path
- Hermes root (`~/.hermes`)
- Which canonical files are present
- Session source (`state.db` or `.jsonl` fallback)

### Files — editing canonical Hermes files

The app provides conflict-aware editing for three files:

| File | Purpose |
|---|---|
| `~/.hermes/memories/USER.md` | User context fed to the agent |
| `~/.hermes/memories/MEMORY.md` | Agent working memory |
| `~/.hermes/SOUL.md` | Agent persona/soul definition |

**Conflict protection flow:**
1. App records the remote file hash at load time
2. Before saving, it re-fetches the remote hash
3. If the file changed on the host since load → save is blocked, edits preserved
4. Prompt: **Reload from Remote** → review diff → re-apply edits → save

### Sessions

- Source: `~/.hermes/state.db` (SQLite, canonical)
- Fallback: `~/.hermes/sessions/*.jsonl` if SQLite unavailable
- Features: search, metadata display, remote deletion, refresh-on-entry

### Usage

Aggregate token dashboard sourced from `~/.hermes/state.db`:
- Total input / output tokens
- Top sessions by token count
- Per-model breakdowns
- Recent session trend

### Skills

Recursive browser for `~/.hermes/skills/**/SKILL.md`:
- Discovers all nested skill files
- Quick filter/search in the list
- Full markdown preview

### Terminal

Embedded SSH terminal with tabs. Uses `/usr/bin/ssh` — the same binary as Terminal.app. Connects to the same host as the active profile.

---

## SSH Configuration Patterns

### Key-based auth setup (if not already done)

```bash
# Generate a key on your Mac if needed
ssh-keygen -t ed25519 -C "hermes-desktop" -f ~/.ssh/id_hermes

# Copy public key to the remote host
ssh-copy-id -i ~/.ssh/id_hermes.pub user@host

# Or manually:
cat ~/.ssh/id_hermes.pub | ssh user@host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# Test — must return without any prompt
ssh -i ~/.ssh/id_hermes user@host echo ok
```

### SSH config for common topologies

```sshconfig
# Tailscale (MagicDNS hostname)
Host hermes-tailscale
  HostName my-machine.tail1234.ts.net
  User alex
  IdentityFile ~/.ssh/id_ed25519

# Jump host / bastion
Host hermes-behind-bastion
  HostName 10.0.0.5
  User ubuntu
  ProxyJump bastion.example.com
  IdentityFile ~/.ssh/id_ed25519

# Same Mac via localhost
Host hermes-local
  HostName localhost
  User alex
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking no
```

> **Note:** Hermes Desktop uses standard `/usr/bin/ssh`. If your setup requires the separate `tailscale ssh` command, use a ProxyCommand instead:

```sshconfig
Host hermes-tailscale-cmd
  HostName my-machine
  User alex
  ProxyCommand tailscale ssh --nc %h %p
```

---

## Building from Source — Key Files

```
hermes-desktop/
├── scripts/
│   └── build-macos-app.sh          # Universal (arm64 + x86_64) build script
├── HermesDesktop/
│   ├── App/
│   │   └── HermesDesktopApp.swift  # SwiftUI @main entry point
│   ├── Views/
│   │   ├── SessionsView.swift
│   │   ├── FilesView.swift
│   │   ├── UsageView.swift
│   │   ├── SkillsView.swift
│   │   └── TerminalView.swift
│   ├── Models/
│   │   └── ConnectionProfile.swift
│   └── SSH/
│       └── SSHClient.swift         # Wraps /usr/bin/ssh process calls
└── HermesDesktop.xcodeproj
```

### Build script usage

```bash
# Full universal build
./scripts/build-macos-app.sh

# Verify the result is a universal binary
lipo -info HermesDesktop.app/Contents/MacOS/HermesDesktop
# → Architectures in the fat file: arm64 x86_64
```

### Xcode build (development)

```bash
# Debug build for current arch
xcodebuild -scheme HermesDesktop -configuration Debug build

# Run tests
xcodebuild -scheme HermesDesktop test
```

---

## Swift Patterns Used in the Codebase

### Running SSH commands (non-interactive)

```swift
import Foundation

func runSSHCommand(
    alias: String,
    command: String,
    completion: @escaping (Result<String, Error>) -> Void
) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
    process.arguments = [
        "-o", "BatchMode=yes",          // never prompt interactively
        "-o", "ConnectTimeout=10",
        alias,
        command
    ]

    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe

    process.terminationHandler = { proc in
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""

        if proc.terminationStatus == 0 {
            completion(.success(output))
        } else {
            completion(.failure(SSHError.nonZeroExit(output)))
        }
    }

    do {
        try process.run()
    } catch {
        completion(.failure(error))
    }
}

// Usage
runSSHCommand(alias: "hermes-pi", command: "python3 --version") { result in
    switch result {
    case .success(let version): print("python3 ok:", version)
    case .failure(let err): print("preflight failed:", err)
    }
}
```

### Reading a remote file with conflict detection

```swift
struct RemoteFile {
    let content: String
    let sha256: String       // hash at load time
}

func fetchRemoteFile(
    alias: String,
    path: String,
    completion: @escaping (Result<RemoteFile, Error>) -> Void
) {
    // Fetch content and hash in one SSH call
    let cmd = "python3 -c \""
        + "import hashlib, pathlib; "
        + "p = pathlib.Path('\(path)').expanduser(); "
        + "b = p.read_bytes(); "
        + "print(hashlib.sha256(b).hexdigest()); "
        + "print(b.decode())\""

    runSSHCommand(alias: alias, command: cmd) { result in
        switch result {
        case .success(let output):
            let lines = output.components(separatedBy: "\n")
            guard let hash = lines.first else {
                completion(.failure(RemoteFileError.parseError)); return
            }
            let content = lines.dropFirst().joined(separator: "\n")
            completion(.success(RemoteFile(content: content, sha256: hash)))
        case .failure(let err):
            completion(.failure(err))
        }
    }
}

func saveRemoteFile(
    alias: String,
    path: String,
    newContent: String,
    originalHash: String,
    completion: @escaping (Result<Void, Error>) -> Void
) {
    // Check current remote hash before writing
    let checkCmd = "python3 -c \""
        + "import hashlib, pathlib; "
        + "p = pathlib.Path('\(path)').expanduser(); "
        + "print(hashlib.sha256(p.read_bytes()).hexdigest())\""

    runSSHCommand(alias: alias, command: checkCmd) { result in
        switch result {
        case .success(let currentHash):
            if currentHash.trimmingCharacters(in: .whitespacesAndNewlines) != originalHash {
                completion(.failure(RemoteFileError.conflict))
                return
            }
            // Safe to write — pipe content via stdin
            let writeCmd = "cat > \(path)"
            // ... write newContent via stdin to the process
            completion(.success(()))
        case .failure(let err):
            completion(.failure(err))
        }
    }
}

enum RemoteFileError: Error {
    case parseError
    case conflict   // remote changed since load — prompt user to reload
}
```

### SwiftUI connection profile model

```swift
import Foundation

struct ConnectionProfile: Identifiable, Codable {
    var id: UUID = UUID()
    var name: String
    var sshAlias: String?       // preferred: uses ~/.ssh/config
    var host: String?
    var user: String?
    var port: Int = 22

    /// The argument passed to /usr/bin/ssh
    var sshTarget: String {
        if let alias = sshAlias, !alias.isEmpty {
            return alias
        }
        let u = user.map { "\($0)@" } ?? ""
        let p = port != 22 ? " -p \(port)" : ""
        return "\(u)\(host ?? "localhost")\(p)"
    }
}
```

---

## Troubleshooting

### `Test` fails: "Authentication failed"

```bash
# Verify key auth works without prompts
ssh -o BatchMode=yes -o ConnectTimeout=10 your-host echo ok

# If it asks for a password, set up key-based auth:
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@host

# Check the SSH agent has the key loaded
ssh-add -l
ssh-add ~/.ssh/id_ed25519
```

### `Test` fails: "python3 not found"

```bash
# On the remote host, check python3 path
which python3
# If it's in a non-standard location like /usr/local/bin, ensure PATH is set
# in ~/.bashrc or ~/.zshrc (non-interactive shells may have a minimal PATH)

# Quick fix: symlink to a standard location
sudo ln -s /usr/local/bin/python3 /usr/bin/python3
```

### `Test` fails: "Host key verification failed"

```bash
# Accept the host key interactively first
ssh your-host
# Type 'yes' when prompted

# Or clear a stale entry
ssh-keygen -R your-host
ssh your-host   # accept fresh key
```

### Sessions view shows no sessions

The app tries `~/.hermes/state.db` first. Check on the remote host:

```bash
ssh your-host "ls -lh ~/.hermes/state.db"
# If missing, check for jsonl fallback:
ssh your-host "ls ~/.hermes/sessions/*.jsonl | head -5"
```

### Conflict error when saving a memory file

The remote file changed after you opened it. In the app:
1. Click **Reload from Remote**
2. The editor will show the current remote content
3. Re-apply your edits manually
4. Save again

### Tailscale: SSH works with `tailscale ssh` but not `ssh`

Hermes Desktop uses `/usr/bin/ssh`. Configure a ProxyCommand in `~/.ssh/config`:

```sshconfig
Host hermes-tailscale
  HostName my-machine
  User alex
  ProxyCommand /usr/local/bin/tailscale nc %h %p
```

Then use `hermes-tailscale` as the SSH alias in the app.

### Monitor live network activity

```bash
# Watch what the app connects to
sudo nettop -p HermesDesktop

# Or use lsof to see open connections
lsof -i -n -P | grep HermesDesktop
```

---

## Key Design Constraints

| Constraint | Reason |
|---|---|
| No gateway API | Remote host is the only source of truth |
| No local file mirror | Avoids stale state and sync conflicts |
| No remote helper service | Zero install footprint on the host |
| Uses `/usr/bin/ssh` | Same path as Terminal — no novel auth surface |
| `BatchMode=yes` on all non-terminal SSH calls | Prevents silent interactive prompt hangs |
| Conflict check before every save | Protects against agent overwriting newer remote state |
```
