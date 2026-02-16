from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import random

app = FastAPI(title="Restaurant Recommender Agent - Demo Mode")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    user_id: str = "user_001"
    location: str = "India"
    visit_history: Optional[List[Dict]] = None

class RestaurantRecommendation(BaseModel):
    recommendations: str
    crew_log: str

# Demo restaurant data for India
DEMO_RESTAURANTS = {
    "Mumbai": [
        {"name": "Bombay Canteen", "cuisine": "Modern Indian", "rating": 4.5, "price": "$$"},
        {"name": "Trishna", "cuisine": "Seafood", "rating": 4.7, "price": "$$$"},
        {"name": "Britannia & Co", "cuisine": "Parsi", "rating": 4.3, "price": "$"},
    ],
    "Delhi": [
        {"name": "Indian Accent", "cuisine": "Contemporary Indian", "rating": 4.8, "price": "$$$$"},
        {"name": "Karim's", "cuisine": "Mughlai", "rating": 4.4, "price": "$"},
        {"name": "Bukhara", "cuisine": "North Indian", "rating": 4.6, "price": "$$$"},
    ],
    "Bangalore": [
        {"name": "Karavalli", "cuisine": "Coastal Indian", "rating": 4.6, "price": "$$$"},
        {"name": "MTR", "cuisine": "South Indian", "rating": 4.5, "price": "$"},
        {"name": "Toit", "cuisine": "Brewpub", "rating": 4.4, "price": "$$"},
    ],
    "India": [
        {"name": "The Spice Route", "cuisine": "Pan-Asian", "rating": 4.5, "price": "$$$"},
        {"name": "Dum Pukht", "cuisine": "Awadhi", "rating": 4.7, "price": "$$$$"},
        {"name": "Wasabi by Morimoto", "cuisine": "Japanese", "rating": 4.6, "price": "$$$$"},
    ]
}

@app.post("/recommend", response_model=RestaurantRecommendation)
async def recommend(request: RecommendationRequest):
    """
    Demo endpoint - returns mock recommendations without requiring API keys
    """
    location = request.location
    
    # Find matching restaurants
    restaurants = DEMO_RESTAURANTS.get(location, DEMO_RESTAURANTS["India"])
    selected = random.sample(restaurants, min(3, len(restaurants)))
    
    # Generate mock recommendation text
    recommendations = f"""
# Restaurant Recommendations for {location}

## Top 3 Personalized Picks:

"""
    
    for i, rest in enumerate(selected, 1):
        recommendations += f"""
### {i}. {rest['name']}
- **Cuisine**: {rest['cuisine']}
- **Rating**: {rest['rating']}/5.0
- **Price Range**: {rest['price']}
- **Why we recommend**: Based on your dining preferences, this restaurant offers an excellent {rest['cuisine'].lower()} experience with consistently high ratings.

"""
    
    recommendations += """
## 🔥 Trending Now:

**Artisan Bakery & Café**
- A new hotspot featuring farm-to-table ingredients and artisanal breads
- Perfect for brunch enthusiasts
- Rating: 4.5/5.0

---
*Note: This is a DEMO response. To get AI-powered personalized recommendations, please configure your WatsonX API credentials in the .env file.*
"""
    
    crew_log = f"""
[DEMO MODE] Multi-Agent Analysis Complete:
✓ User Profile Agent: Analyzed dining preferences
✓ RAG Matcher: Retrieved top candidates from database  
✓ Recommendation Agent: Selected best matches
✓ Trend Analyzer: Identified trending options

Location: {location}
Restaurants Analyzed: {len(restaurants)}
Final Recommendations: 3
"""
    
    return RestaurantRecommendation(
        recommendations=recommendations,
        crew_log=crew_log
    )

@app.get("/")
def read_root():
    return {
        "status": "Backend is running in DEMO MODE",
        "message": "Configure WatsonX API credentials for full AI-powered recommendations"
    }
