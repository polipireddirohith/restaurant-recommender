import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Check for Gemini key
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_key or gemini_key.startswith("your_"):
        print("=" * 80)
        print("🎭 RUNNING IN DEMO MODE")
        print("=" * 80)
        print("No valid Gemini API key found. Using demo backend with mock data.")
        print("To use the full AI-powered version, update your .env file with:")
        print("  - GEMINI_API_KEY")
        print("=" * 80)
        uvicorn.run("app.main_demo:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print("=" * 80)
        print("🤖 RUNNING IN AI MODE (Google Gemini)")
        print("=" * 80)
        print("Gemini credentials detected. Using Gemini-powered backend.")
        print("=" * 80)
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
