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
    allow_origins=["*"],
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
    watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    watsonx_apikey = os.getenv("WATSONX_APIKEY")
    watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", "skills-network")
    serper_api_key = os.getenv("SERPER_API_KEY")

    if not watsonx_apikey:
        print("Warning: WATSONX_APIKEY not set. RAG and LLM might fail.")
    
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
        search_tool = SerperDevTool()
    else:
        print("Warning: SERPER_API_KEY not set. Food Trend Agent will be disabled.")

    # Setup RAG
    print("Setting up RAG Retriever...")
    try:
        retriever_instance = setup_rag(api_key=watsonx_apikey, project_id=watsonx_project_id, url=watsonx_url)
        # Create Tool instance and inject retriever
        rag_retriever = RAG_Retriever()
        rag_retriever.retriever = retriever_instance
        retriever_tool = rag_retriever
        print("RAG Retriever setup complete.")
    except Exception as e:
        print(f"Failed to setup RAG: {e}")

    # Setup LLM
    print("Setting up LLM...")
    try:
        # Using Watsonx LLM via CrewAI LLM wrapper if compatible or LangChain
        # The notebook used CrewAI's LLM class.
        llm = LLM(
            model="watsonx/meta-llama/llama-3-3-70b-instruct",
            base_url=watsonx_url,
            project_id=watsonx_project_id,
            max_tokens=2000,
            api_key=watsonx_apikey
        )
        print("LLM setup complete.")
    except Exception as e:
         print(f"Failed to setup LLM: {e}")


@app.post("/recommend", response_model=RestaurantRecommendation)
async def recommend(request: RecommendationRequest):
    global llm, retriever_tool, search_tool

    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized.")
    
    # 1. Image Analysis (if images are provided in visit history)
    # For now, we assume visit_history has pre-processed descriptions or we reuse the dummy data
    # In a real app, we would process images here using restaurant_image_analysis
    
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

    # Re-import Task to avoid circular issues or just define here
    from crewai import Task
    from app.models import UserProfile

    # User Profile Task
    user_profile_task = Task(
        description=task_config['user_profile_task']['description'],
        expected_output=task_config['user_profile_task']['expected_output'],
        agent=user_profile_builder,
        output_pydantic=UserProfile
    )

    # Coarse RAG Task
    coarse_RAG_match_task = Task(
        description=task_config['coarse_RAG_match_task']['description'],
        expected_output=task_config['coarse_RAG_match_task']['expected_output'],
        agent=coarse_RAG_matcher,
        context=[user_profile_task]
    )

    tasks = [user_profile_task, coarse_RAG_match_task]
    agents = [user_profile_builder, coarse_RAG_matcher]

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
    context_tasks = [user_profile_task, coarse_RAG_match_task]
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

    # 5. Kickoff
    inputs = {
        "visit_history": visit_history if visit_history else [],
        "location": request.location
    }
    
    try:
        result = crew.kickoff(inputs=inputs)
        return RestaurantRecommendation(
            recommendations=str(result),
            crew_log="Crew execution completed." # Capture logs if possible
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Backend is running"}
