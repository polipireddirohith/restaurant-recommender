from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class RetrieverToolInput(BaseModel):
    """Input schema for RAG Retriever Tool."""
    user_profile: str = Field(..., description="The string version of the user profile dictionary")

class UserProfile(BaseModel):
    preferred_cuisines: Dict[str, float] = Field(..., description="A dictionary of cuisines and a preference score (0-1) for each.")
    price_tier_preference: int = Field(..., description="The user's preferred price tier (e.g., 0 for the lowest, 1 for $, 2 for $$, etc.).")
    avg_rating_preference: float = Field(..., description="The average rating the user prefers in restaurants (e.g., 4.5/5).")
    dining_environment_preference: str = Field(..., description="A short summary of the user's preferred dining environment.")
    summary: str = Field(..., description="A short text summary of the overall user profile.")

class RecommendationRequest(BaseModel):
    user_id: str = "user_001"
    location: str = "India"
    # Optional visit history to override default
    visit_history: Optional[List[Dict]] = None

class RestaurantRecommendation(BaseModel):
    recommendations: str
    crew_log: str
