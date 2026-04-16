import requests

SUBRIFF_API = "https://subriff.com/Home/GetSubreddits"


def scrape_fastest_growing(min_size=1000, limit=20):
    """Find fastest-growing subreddits via Subriff API."""
    params = {
        "minSize": min_size,
        "sortBy": "weeklyGrowth",
        "sortDir": "desc",
        "page": 1,
        "pageSize": limit,
        "sfw": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json",
        "Referer": "https://subriff.com/",
    }

    resp = requests.get(SUBRIFF_API, params=params, headers=headers, timeout=45)
    resp.raise_for_status()
    data = resp.json()

    subs = []
    for s in data.get("subreddits", []):
        subs.append({
            "name": s.get("displayName", ""),
            "subscribers": s.get("subscribers", 0),
            "description": (s.get("publicDescription", "") or "")[:200],
            "daily_growth": s.get("dailyGrowth", 0),
            "weekly_growth": s.get("weeklyGrowth", 0),
            "weekly_growth_pct": round(s.get("weeklyGrowthPercentage", 0), 1),
            "monthly_growth": s.get("monthlyGrowth", 0),
            "created": s.get("subredditCreatedUtc", ""),
            "url": f"https://reddit.com{s.get('url', '')}",
        })

    return subs


def scrape_new_and_growing(min_size=500, limit=15):
    """Find recently created subreddits that are growing fast."""
    params = {
        "minSize": min_size,
        "sortBy": "dailyGrowthPercentage",
        "sortDir": "desc",
        "page": 1,
        "pageSize": limit,
        "sfw": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json",
        "Referer": "https://subriff.com/",
    }

    resp = requests.get(SUBRIFF_API, params=params, headers=headers, timeout=45)
    resp.raise_for_status()
    data = resp.json()

    subs = []
    for s in data.get("subreddits", []):
        subs.append({
            "name": s.get("displayName", ""),
            "subscribers": s.get("subscribers", 0),
            "description": (s.get("publicDescription", "") or "")[:200],
            "daily_growth_pct": round(s.get("dailyGrowthPercentage", 0), 1),
            "weekly_growth_pct": round(s.get("weeklyGrowthPercentage", 0), 1),
            "created": s.get("subredditCreatedUtc", ""),
            "url": f"https://reddit.com{s.get('url', '')}",
        })

    return subs


if __name__ == "__main__":
    print("--- Fastest Growing (by weekly subscribers) ---")
    fast = scrape_fastest_growing(min_size=5000, limit=10)
    for s in fast:
        print(f"  r/{s['name']} ({s['subscribers']:,} subs, +{s['weekly_growth']:,}/wk, {s['weekly_growth_pct']}%)")
        if s["description"]:
            print(f"    {s['description'][:80]}")

    print("\n--- New & Exploding (by daily growth %) ---")
    new = scrape_new_and_growing(min_size=1000, limit=10)
    for s in new:
        print(f"  r/{s['name']} ({s['subscribers']:,} subs, +{s['daily_growth_pct']}%/day)")
        if s["description"]:
            print(f"    {s['description'][:80]}")
