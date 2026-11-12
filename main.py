import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from offline_places import load_places, build_itinerary

app = FastAPI(title="OdysseyAi AI Service")

pipe = joblib.load("sentiment_model.joblib")

try:
    places_df = load_places("places.csv")
except Exception as error:
    print("Could not load places.csv:", error)
    places_df = None


class Hotel(BaseModel):
    name: str
    reviews: list[str]

class HotelList(BaseModel):
    hotels: list[Hotel]


def hotel_stars(reviews):
    reviews = [r for r in reviews if r.strip()]
    if len(reviews) == 0:
        return None
    probs = pipe.predict_proba(reviews)
    stars_each = probs @ pipe.classes_
    return float(stars_each.mean())


@app.get("/health")
def health():
    return {"status": "ok", "placesLoaded": places_df is not None}


@app.post("/rank-hotels")
def rank_hotels(body: HotelList):
    if len(body.hotels) == 0:
        raise HTTPException(status_code=400, detail="no hotels given")

    scored = []
    unscored = []
    for hotel in body.hotels:
        stars = hotel_stars(hotel.reviews)
        if stars is None:
            unscored.append({"name": hotel.name, "avgExpectedStars": None})
        else:
            scored.append({"name": hotel.name, "avgExpectedStars": round(stars, 2)})

    scored.sort(key=lambda h: h["avgExpectedStars"], reverse=True)
    ranked = scored + unscored
    for position, hotel in enumerate(ranked, start=1):
        hotel["rank"] = position
    return {"ranking": ranked}


class OfflineRequest(BaseModel):
    destination: str
    preferences: str = ""
   


@app.post("/offline-itinerary")
def offline_itinerary(body: OfflineRequest):
    if places_df is None:
        raise HTTPException(status_code=503, detail="places.csv could not be loaded")
    try:
        return build_itinerary(places_df, body.destination,
                               body.preferences)
    except LookupError as error:                 
        raise HTTPException(status_code=404, detail=str(error))