from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv

from crewai import Crew, Process, LLM
from crewai_tools import SerperDevTool

from app.models import RecommendationRequest, RestaurantRecommendation
from app.tools import setup_rag, RAG_Retriever, restaurant_image_analysis
from app.agents import create_agents
from app.tasks import load_task_config # Adjusted import
from app.data_loader import load_default_history

# Load environment variables
load_dotenv()

app = FastAPI(title="Restaurant Recommender Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for rigorous initialization
retriever_tool = None
search_tool = None
llm = None
rag_retriever = None

@app.on_event("startup")
async def startup_event():
    global retriever_tool, search_tool, llm, rag_retriever
    
    # Check credentials
    gemini_apikey = os.getenv("GEMINI_API_KEY")
    serper_api_key = os.getenv("SERPER_API_KEY")

    # Check if API key is valid (not placeholder)
    if not gemini_apikey or gemini_apikey.startswith("your_"):
        print("=" * 80)
        print("ERROR: GEMINI_API_KEY not configured!")
        print(" Please update the .env file with your actual Google Gemini API key.")
        print("=" * 80)
        gemini_apikey = None
    
    if serper_api_key and not serper_api_key.startswith("your_"):
        os.environ["SERPER_API_KEY"] = serper_api_key
        try:
            search_tool = SerperDevTool()
            print("Serper search tool initialized.")
        except Exception as e:
            print(f"Warning: Failed to initialize Serper tool: {e}")
    else:
        print("Warning: SERPER_API_KEY not set. Food Trend Agent will be disabled.")

    # Setup RAG
    print("[V2] Setting up RAG Retriever...")
    if gemini_apikey:
        try:
            retriever_instance = setup_rag(api_key=gemini_apikey)
            if retriever_instance:
                # Create Tool instance and inject retriever
                rag_retriever = RAG_Retriever()
                rag_retriever.retriever = retriever_instance
                retriever_tool = rag_retriever
                print("RAG Retriever setup complete.")
            else:
                print("RAG setup failed but server continuing.")
                retriever_tool = None
        except Exception as e:
            print(f"Failed to setup RAG: {e}")
            retriever_tool = None
    else:
        print("Skipping RAG setup - no API key")

    # Setup LLM
    print("Setting up LLM...")
    if gemini_apikey:
        import google.generativeai as genai
        genai.configure(api_key=gemini_apikey)
        candidates = []
        # Add any models actually found in the list that look like flash/pro
        try:
            available = [m.name.split("/")[-1] for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
            print(f"Discovered generative models: {available}")
            import sys
            sys.stdout.flush()
            # Prefer flash models
            for m in available:
                if "flash" in m and "lite" not in m:
                    candidates.append(m)
            # Then pro models
            for m in available:
                if "pro" in m and m not in candidates:
                    candidates.append(m)
            # Then anything else
            for m in available:
                if m not in candidates:
                    candidates.append(m)
        except Exception as e:
            print(f"Note: Could not list models: {e}")
            import sys
            sys.stdout.flush()
            candidates = ["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-pro"]

        llm = None
        for model_name in candidates:
            try:
                print(f"Trying LLM model: {model_name}...")
                import sys
                sys.stdout.flush()
                test_llm = LLM(model=f"gemini/{model_name}", api_key=gemini_apikey)
                # Success if no immediate error (CrewAI LLM doesn't validate on init usually)
                llm = test_llm
                print(f"Selected LLM model: {model_name}")
                sys.stdout.flush()
                break
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue
        
        if not llm:
            print("FAILED to find any working LLM model.")
    else:
        print("Skipping LLM setup - no API key")
        llm = None


@app.post("/recommend", response_model=RestaurantRecommendation)
async def recommend(request: RecommendationRequest):
    global llm, retriever_tool, search_tool

    if not llm:
        raise HTTPException(
            status_code=503, 
            detail="Service not available: Google Gemini API key not configured. Please update the .env file with valid GEMINI_API_KEY."
        )
    
    # 1. Image Analysis (if images are provided in visit history)
    visit_history = request.visit_history
    if not visit_history:
        print("Loading default visit history...")
        try:
             # Ideally load from data_loader
             from app.data_loader import load_default_history
             visit_history = load_default_history()
        except Exception as e:
             print(f"Failed to load default history: {e}")
             visit_history = []

    # 2. Create Agents
    tools_map = {
        'retriever_tool': retriever_tool,
        'search_tool': search_tool
    }
    
    user_profile_builder, coarse_RAG_matcher, restaurant_recommendation_expert, food_trend_researcher = create_agents(llm, tools_map)

    # 3. Create Tasks
    # We need to recreate tasks structure dynamically
    from app.tasks import load_task_config
    task_config = load_task_config()

    from crewai import Task
    from app.models import UserProfile

    # User Profile Task
    user_profile_task = Task(
        description=task_config['user_profile_task']['description'],
        expected_output=task_config['user_profile_task']['expected_output'],
        agent=user_profile_builder,
        output_pydantic=UserProfile
    )

    tasks = [user_profile_task]
    agents = [user_profile_builder]

    # Coarse RAG Task
    coarse_RAG_match_task = None
    if retriever_tool:
        coarse_RAG_match_task = Task(
            description=task_config['coarse_RAG_match_task']['description'],
            expected_output=task_config['coarse_RAG_match_task']['expected_output'],
            agent=coarse_RAG_matcher,
            context=[user_profile_task]
        )
        tasks.append(coarse_RAG_match_task)
        agents.append(coarse_RAG_matcher)
    else:
        print("Skipping Coarse RAG Match Task - retriever tool not available")

    # Food Trend Task
    food_trend_task = None
    if food_trend_researcher:
        food_trend_task = Task(
            description=task_config['food_trend_task']['description'],
            expected_output=task_config['food_trend_task']['expected_output'],
            agent=food_trend_researcher
        )
        tasks.append(food_trend_task)
        agents.append(food_trend_researcher)

    # Recommendation Task
    context_tasks = [user_profile_task]
    if coarse_RAG_match_task:
        context_tasks.append(coarse_RAG_match_task)
    if food_trend_task:
        context_tasks.append(food_trend_task)

    restaurant_recommendation_task = Task(
        description=task_config['restaurant_recommendation_task']['description'],
        expected_output=task_config['restaurant_recommendation_task']['expected_output'],
        agent=restaurant_recommendation_expert,
        context=context_tasks
    )
    tasks.append(restaurant_recommendation_task)
    agents.append(restaurant_recommendation_expert)

    # 4. Create Crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    print(f"Kicking off Crew with location: {request.location}")
    try:
        # 5. Kickoff
        inputs = {
            "visit_history": visit_history if visit_history else [],
            "location": request.location
        }
        
        result = crew.kickoff(inputs=inputs)
        return RestaurantRecommendation(
            recommendations=str(result),
            crew_log="Crew execution completed."
        )
    except Exception as e:
        print(f"ERROR during crew kickoff: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Crew execution failed",
                "message": str(e),
                "suggestion": "Check if your Gemini API key has access to the specified model."
            }
        )

@app.get("/")
def read_root():
    return {"status": "Backend is running with Google Gemini"}
