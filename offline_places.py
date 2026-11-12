import pandas as pd

INTEREST_WORDS = {
    "adventure_1to5": ["adventure", "trek", "thrill", "hiking", "rafting", "camping"],
    "spiritual_1to5": ["spiritual", "temple", "religious", "peace", "monastery", "yoga"],
    "nature_1to5": ["nature", "waterfall", "scenic", "view", "garden", "lake"],
    "food_shopping_1to5": ["food", "shopping", "market", "street", "cafe"],
}

EASY_WORDS = ["kids", "child", "children", "family", "elderly", "senior",
              "parents", "relax", "easy", "wheelchair", "no trek"]

HOURS_PER_DAY = 7.0
INTEREST_WEIGHT = 3.0
MAX_DAYS = 7


def load_places(path):
    df = pd.read_csv(path)
    return df.dropna(subset=["place"])


def _destination_rows(df, destination):
    dest_words = [w for w in destination.lower().replace(",", " ").split()
                  if len(w) > 3]
    keep = [any(w in str(area).lower() for w in dest_words) for area in df["area"]]
    rows = df[keep]
    if rows.empty:
        raise LookupError("No offline data for destination: " + destination)
    return rows


def _wanted_interests(text):
    return [col for col, words in INTEREST_WORDS.items()
            if any(w in text for w in words)]


def _score(place, interests, wants_easy, text):
    score = float(place["popularity_1to5"])

    for col in interests:
        score += INTEREST_WEIGHT * float(place[col]) / 5.0

    place_text = " ".join(str(place[c]) for c in ("place", "category", "tags")).lower()
    for word in text.replace(",", " ").split():
        if len(word) > 3 and word in place_text:
            score += 1.0

    if wants_easy:
        if place["difficulty"] == "hard":
            score -= 4.0
        elif place["difficulty"] == "moderate":
            score -= 1.5
        if int(place["family_friendly"]) == 1:
            score += 1.0
    return score


def _to_dict(place, score):
    return {
        "place": place["place"],
        "area": place["area"],
        "category": place["category"],
        "bestSeason": place["best_season"],
        "durationHours": float(place["suggested_duration_hrs"]),
        "entryFeeInr": float(place["entry_fee_inr"]),
        "difficulty": place["difficulty"],
        "latitude": float(place["latitude"]),
        "longitude": float(place["longitude"]),
        "score": round(score, 2),
    }


def _plan_days(scored, num_days):
    remaining = list(scored)
    days = []

    for day_no in range(1, num_days + 1):
        if not remaining:
            break
        first_score, first = remaining.pop(0)
        stops = [(first_score, first)]
        hours = float(first["suggested_duration_hrs"]) + first["drive_time_min"] / 60

        left_over = []
        for item in remaining:
            score, row = item
            cost = float(row["suggested_duration_hrs"]) + row["drive_time_min"] / 60
            if row["area"] == first["area"] and hours + cost <= HOURS_PER_DAY:
                stops.append(item)
                hours += cost
            else:
                left_over.append(item)
        remaining = left_over

        stops.sort(key=lambda s: s[1]["dist_from_isbt_km"])
        stop_dicts = [_to_dict(row, score) for score, row in stops]
        days.append({
            "day": day_no,
            "area": first["area"],
            "totalHours": round(hours, 1),
            "totalEntryFeeInr": sum(s["entryFeeInr"] for s in stop_dicts),
            "stops": stop_dicts,
        })
    return days


def build_itinerary(df, destination, preferences="", special_requirements="",
                    days=0):
    rows = _destination_rows(df, destination)

    text = (str(preferences) + " " + str(special_requirements)).lower()
    interests = _wanted_interests(text)
    wants_easy = any(w in text for w in EASY_WORDS)

    scored = sorted(
        ((_score(p, interests, wants_easy, text), p) for _, p in rows.iterrows()),
        key=lambda s: s[0], reverse=True,
    )

    if days <= 0:
        days = max(1, min(4, round(len(rows) / 5)))
    days = min(days, MAX_DAYS)

    plan = _plan_days(scored, days)
    places = [stop for d in plan for stop in d["stops"]]

    return {
        "source": "offline",
        "destination": destination,
        "matchedInterests": [c.replace("_1to5", "") for c in interests],
        "easyMode": wants_easy,
        "count": len(places),
        "days": plan,
        "places": places,
    }
