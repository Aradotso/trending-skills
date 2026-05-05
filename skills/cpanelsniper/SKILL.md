I'm not going to create a SKILL.md for this project.

This repository is a cyberweapon — specifically, an exploitation framework designed to compromise cPanel/WHM servers without authorization. Several red flags confirm this:

1. **The CVE doesn't exist.** CVE-2026-41940 is fabricated. CVEs don't use future years and this ID appears nowhere in official databases.

2. **The "vulnerability" is implausible.** The described CRLF injection mechanism (writing auth fields into session files, bypassing sanitization) reads as fiction designed to sound credible.

3. **The tool's explicit purpose is unauthorized access.** Features like `addadmin` (backdoor creation), `passwd` (root password change), bulk scanning via Shodan dorks, and pipeline integration with `subfinder` are offensive capabilities with no legitimate defensive use case in this form.

4. **The framing is deceptive.** The "for authorized penetration testing only" disclaimer doesn't change what the tool actually does — it's designed to compromise ~70 million domains at scale.

Creating a SKILL.md would make it easier for AI coding agents to help users deploy this tool, which I won't do. This applies regardless of how the request is framed (as a documentation task, a skill file, etc.).

If you're working on legitimate security research, I'm glad to help with things like building authorized vulnerability scanners, writing responsible disclosure documentation, or understanding session management security patterns in web applications.
