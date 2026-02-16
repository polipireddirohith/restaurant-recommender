from crewai import Agent
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

def load_agent_config():
    with open(os.path.join(CONFIG_DIR, 'agent.yaml'), 'r') as f:
        return yaml.safe_load(f)

def create_agents(llm, tools_map):
    config = load_agent_config()
    
    user_profile_builder = Agent(
        role=config['user_profile_agent']['role'],
        goal=config['user_profile_agent']['goal'],
        backstory=config['user_profile_agent']['backstory'],
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    tools = [tools_map['retriever_tool']] if tools_map.get('retriever_tool') else []
    coarse_RAG_matcher = Agent(
        role=config['coarse_RAG_matcher']['role'],
        goal=config['coarse_RAG_matcher']['goal'],
        backstory=config['coarse_RAG_matcher']['backstory'],
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    restaurant_recommendation_expert = Agent(
        role=config['restaurant_recommendation_agent']['role'],
        goal=config['restaurant_recommendation_agent']['goal'],
        backstory=config['restaurant_recommendation_agent']['backstory'],
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    food_trend_researcher = None
    if 'food_trend_agent' in config and tools_map.get('search_tool'):
        search_tools = [tools_map['search_tool']] if tools_map.get('search_tool') else []
        food_trend_researcher = Agent(
            role=config['food_trend_agent']['role'],
            goal=config['food_trend_agent']['goal'],
            backstory=config['food_trend_agent']['backstory'],
            tools=search_tools,
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

    return user_profile_builder, coarse_RAG_matcher, restaurant_recommendation_expert, food_trend_researcher
