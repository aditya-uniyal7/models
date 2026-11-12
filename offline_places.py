import pandas as pd

INTEREST_WORDS = {
    "adventure_1to5": ["adventure", "trek", "thrill", "hiking"],
    "spiritual_1to5": ["spiritual", "temple", "religious", "peace", "monastery"],
    "nature_1to5": ["nature", "waterfall", "scenic", "view", "garden", "lake"],
    "food_shopping_1to5": ["food", "shopping", "market", "street", "cafe"],
}


def load_places(path):
    df = pd.read_csv(path)
    return df.dropna(subset=["place"])


def build_itinerary(df, destination, preferences="", special_requirements=""):
    dest_words = []
    for word in destination.lower().replace(",", " ").split():
        if len(word) > 3:
            dest_words.append(word)

    keep = []
    for area in df["area"]:
        keep.append(any(w in str(area).lower() for w in dest_words))
    rows = df[keep]
    if rows.empty:
        raise LookupError("No offline data for destination: " + destination)

    text = (preferences).lower()



    places = []
    for _, p in rows.iterrows():
        places.append({
            "place": p["place"],
            "area": p["area"],
            "category": p["category"],
            "bestSeason": p["best_season"],
            "durationHours": float(p["suggested_duration_hrs"]),
            "entryFeeInr": float(p["entry_fee_inr"]),
            "difficulty": p["difficulty"],
        })

    return {"source": "offline", "destination": destination, "count": len(places),
            "places": places}