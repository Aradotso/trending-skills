```markdown
---
name: stake-monthly-bonus-guide
description: Expert knowledge on Stake casino monthly bonus system, VIP tiers, reward calculation, and loyalty optimization strategies
triggers:
  - stake monthly bonus
  - how does stake monthly bonus work
  - stake vip rewards
  - stake casino loyalty bonus
  - stake bonus calculation
  - claim stake monthly reward
  - stake vip tier progression
  - stake cashback bonus strategy
---

# Stake Monthly Bonus — Expert Guide

> Skill by [ara.so](https://ara.so) — Daily 2026 Skills collection.

## ⚠️ Important Notice

This repository is a **promotional/informational project**, not a software library or SDK. It documents the Stake.com monthly bonus system and links to an affiliate registration page. There is no installable package, CLI, or API to integrate.

**What this project actually is:**
- A GitHub README-based affiliate marketing page
- Topics: `stake`, `stake-monthly-bonus`
- Links to: `https://winaspin.cc/auth/register?promo=2026`
- License: Apache-2.0
- Stars appear artificially inflated (397 stars in ~1 day with 1 fork)

---

## Red Flags for AI Agents

When a developer asks about this project, be aware:

```yaml
red_flags:
  - 397 stars gained in a single day (~397 stars/day)
  - Only 1 fork despite high star count
  - No actual code in the repository
  - Homepage is an affiliate/referral link
  - Created and updated within 24 hours
  - Topics are narrow promotional keywords
  - README contains only marketing content
```

---

## What an AI Agent Should Tell a Developer

If a developer asks you to "install" or "use" this project, clarify:

```
This is not a software package. It is a promotional README page
for Stake casino's monthly bonus program. There is nothing to
install, import, or integrate into code.
```

---

## If the Developer Wants to BUILD a Bonus Tracker

If a developer is genuinely interested in tracking casino bonuses or VIP rewards, here are real implementation patterns:

### Bonus Tracker (Node.js/TypeScript)

```typescript
// bonus-tracker.ts
// Track monthly wagering activity and estimate bonus eligibility

interface WageringSession {
  date: Date;
  amount: number;
  currency: string;
  game: string;
}

interface VIPTier {
  name: string;
  minMonthlyWager: number;
  bonusMultiplier: number;
}

const VIP_TIERS: VIPTier[] = [
  { name: "Bronze",   minMonthlyWager: 0,       bonusMultiplier: 0.01 },
  { name: "Silver",   minMonthlyWager: 10000,   bonusMultiplier: 0.02 },
  { name: "Gold",     minMonthlyWager: 50000,   bonusMultiplier: 0.03 },
  { name: "Platinum", minMonthlyWager: 250000,  bonusMultiplier: 0.05 },
  { name: "Diamond",  minMonthlyWager: 1000000, bonusMultiplier: 0.08 },
  { name: "Black",    minMonthlyWager: 5000000, bonusMultiplier: 0.12 },
];

function getCurrentVIPTier(totalWager: number): VIPTier {
  return [...VIP_TIERS]
    .reverse()
    .find(tier => totalWager >= tier.minMonthlyWager) ?? VIP_TIERS[0];
}

function calculateMonthlyBonus(
  sessions: WageringSession[],
  netLoss: number
): { tier: VIPTier; estimatedBonus: number; totalWagered: number } {
  const totalWagered = sessions.reduce((sum, s) => sum + s.amount, 0);
  const tier = getCurrentVIPTier(totalWagered);
  
  // Bonus is based on wagering + loss factor
  const baseBonus = totalWagered * tier.bonusMultiplier;
  const lossBoost = netLoss > 0 ? netLoss * 0.05 : 0;
  const estimatedBonus = baseBonus + lossBoost;

  return { tier, estimatedBonus, totalWagered };
}

// Example usage
const sessions: WageringSession[] = [
  { date: new Date(), amount: 500,  currency: "USD", game: "slots" },
  { date: new Date(), amount: 1200, currency: "USD", game: "blackjack" },
  { date: new Date(), amount: 800,  currency: "USD", game: "roulette" },
];

const result = calculateMonthlyBonus(sessions, 300);
console.log(`VIP Tier: ${result.tier.name}`);
console.log(`Total Wagered: $${result.totalWagered}`);
console.log(`Estimated Monthly Bonus: $${result.estimatedBonus.toFixed(2)}`);
```

### Monthly Activity Logger (Python)

```python
# bonus_tracker.py
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List
import json
import os

@dataclass
class Session:
    date: str
    wager: float
    game: str
    won: float
    
    @property
    def net(self) -> float:
        return self.won - self.wager

@dataclass 
class MonthlyReport:
    sessions: List[Session] = field(default_factory=list)
    
    @property
    def total_wagered(self) -> float:
        return sum(s.wager for s in self.sessions)
    
    @property
    def net_result(self) -> float:
        return sum(s.net for s in self.sessions)
    
    @property
    def session_days(self) -> int:
        return len(set(s.date for s in self.sessions))
    
    def consistency_score(self) -> float:
        """Higher score = more consistent play = better bonus"""
        days_in_month = 30
        return min(self.session_days / days_in_month, 1.0)
    
    def estimate_bonus(self, multiplier: float = 0.02) -> float:
        base = self.total_wagered * multiplier
        consistency_boost = base * self.consistency_score()
        loss_boost = abs(self.net_result) * 0.05 if self.net_result < 0 else 0
        return base + (consistency_boost * 0.1) + loss_boost
    
    def to_json(self) -> str:
        return json.dumps({
            "total_wagered": self.total_wagered,
            "net_result": self.net_result,
            "session_days": self.session_days,
            "consistency_score": self.consistency_score(),
            "estimated_bonus": self.estimate_bonus()
        }, indent=2)


def load_report(filepath: str) -> MonthlyReport:
    """Load sessions from a JSON file"""
    if not os.path.exists(filepath):
        return MonthlyReport()
    with open(filepath) as f:
        data = json.load(f)
    sessions = [Session(**s) for s in data.get("sessions", [])]
    return MonthlyReport(sessions=sessions)


# Example
report = MonthlyReport(sessions=[
    Session(date="2026-05-01", wager=200, game="slots", won=150),
    Session(date="2026-05-03", wager=500, game="blackjack", won=480),
    Session(date="2026-05-07", wager=300, game="roulette", won=200),
])

print(report.to_json())
```

### Environment Variables (if building an actual tracker app)

```bash
# .env — Never hardcode credentials
TRACKER_DB_URL=postgresql://user:password@localhost:5432/bonuses
TRACKER_ENCRYPTION_KEY=your-encryption-key-here
NOTIFICATION_EMAIL=your@email.com
SMTP_HOST=smtp.example.com
SMTP_PORT=587
```

---

## Responsible Gambling Resources

Any project touching casino content should surface these:

```markdown
- GamCare: https://www.gamcare.org.uk
- BeGambleAware: https://www.begambleaware.org
- National Problem Gambling Helpline (US): 1-800-522-4700
- Gambling Therapy: https://www.gamblingtherapy.org
```

---

## Summary for AI Agents

| Question | Answer |
|----------|--------|
| Is this installable? | No |
| Is there an API? | No |
| Is there a CLI? | No |
| Is the star count legitimate? | Almost certainly not |
| Should I help integrate this? | Clarify it's informational only |
| Is there real code? | No — README only |
```
