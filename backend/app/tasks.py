from crewai import Task
import yaml
import os
from app.models import UserProfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

def load_task_config():
    with open(os.path.join(CONFIG_DIR, 'tasks.yaml'), 'r') as f:
        return yaml.safe_load(f)

def create_tasks(agents, tasks_map):
    """
    agents: dict of agent objects
    tasks_map: optional context input
    """
    config = load_task_config()

    user_profile_task = Task(
        description=config['user_profile_task']['description'],
        expected_output=config['user_profile_task']['expected_output'],
        agent=agents['user_profile_builder'],
        output_pydantic=UserProfile
    )

    coarse_RAG_match_task = Task(
        description=config['coarse_RAG_match_task']['description'],
        expected_output=config['coarse_RAG_match_task']['expected_output'],
        agent=agents['coarse_RAG_matcher'],
        context=[user_profile_task], # Replaces depends_on for direct data flow often
        input_data=lambda outputs:{
            'user_profile': outputs['user_profile_task'].raw
        } if 'user_profile_task' in outputs else {} # Needs careful handling of outputs if using lambda
    )
    # Note: In recent CrewAI versions, `context` is preferred over `depends_on` when passing data.
    # The lambda input_data approach used in the notebook is slightly older or specific.
    # We will stick to the notebook style if possible, but context is more robust.
    # However, `input_data` lambda receives `outputs` which is a dict of task outputs.
    # The key is the task object or description? Use context for safety.
    
    # Redefining coarse_RAG_match_task without lambda for simplicity if possible, or keeping it if needed.
    # The notebook used:
    # input_data=lambda outputs:{ 'user_profile': outputs['user_profile_task'].raw }
    # This implies there is a dependency on 'user_profile_task'.
    
    # Let's use context=[user_profile_task] and let the agent handle context variable.
    # But the notebook specifically maps it.

    # We will try to mirror the notebook structure but `depends_on` expects task objects.
    
    coarse_RAG_match_task = Task(
       description=config['coarse_RAG_match_task']['description'],
       expected_output=config['coarse_RAG_match_task']['expected_output'],
       agent=agents['coarse_RAG_matcher'],
       context=[user_profile_task] 
    )

    food_trend_task = None
    if agents.get('food_trend_researcher'):
        food_trend_task = Task(
            description=config['food_trend_task']['description'],
            expected_output=config['food_trend_task']['expected_output'],
            agent=agents['food_trend_researcher']
        )
    
    # Recommendation Task
    context_tasks = [user_profile_task, coarse_RAG_match_task]
    if food_trend_task:
        context_tasks.append(food_trend_task)

    restaurant_recommendation_task = Task(
        description=config['restaurant_recommendation_task']['description'],
        expected_output=config['restaurant_recommendation_task']['expected_output'],
        agent=agents['restaurant_recommendation_expert'],
        context=context_tasks
    )

    return user_profile_task, coarse_RAG_match_task, restaurant_recommendation_task, food_trend_task
