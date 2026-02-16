import os
import json
import requests
import base64
from io import BytesIO
from typing import List, Dict, Type

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ibm import WatsonxEmbeddings
from crewai.tools import BaseTool, tool
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai import Credentials
from app.models import RetrieverToolInput  # Import from local package
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def setup_rag(api_key=None, project_id=None, url=None):
    """
    Initializes the RAG retriever tool.
    """
    if not url:
        raise ValueError("URL for Watsonx must be provided.")
    if not project_id:
        raise ValueError("Project ID for Watsonx must be provided.")

    db_path = os.path.join(DATA_DIR, 'db.txt')
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}. Using empty DB.")
        restaurant_db = {'restaurants': [], 'cafes': [], 'bakeries': []}
    else:
        with open(db_path, 'r', encoding='utf-8') as f:
            restaurant_db = json.load(f)

    # Convert restaurant data to Document objects
    doc_id = 0
    documents = []
    sources = list(restaurant_db.keys())  # restaurants, cafes, bakeries

    for source in sources:
        for j in range(len(restaurant_db[source])):
            shop = restaurant_db[source][j]
            signature_dishes = ''
            for i, dish in enumerate(shop.get('signature_items', [])):
                 signature_dishes += f'Dish {i+1}: {dish}.\n'
            
            review_contents = ''
            for i, title in enumerate(shop.get('review_titles', [])):
                 text = shop['reviews'][i] if i < len(shop.get('reviews', [])) else ""
                 review_contents += f'Review {i+1}. Title: {title}. Text: {text}.\n'
            
            # Use get to avoid KeyErrors
            label = shop.get('label', 'Unknown')
            stype = shop.get('type', 'Unknown')
            location = shop.get('location', 'Unknown')
            rating = shop.get('rating', 0)
            price_range = shop.get('price_range', '')
            short_desc = shop.get('short_description', '')
            
            content = (
                f'Shop name: {label}. ' + 
                f'Shop type: {stype}. ' + 
                f'Shop location: {location}. ' +
                f'Shop rating: {rating}. ' +
                f'Shop price_range: {len(price_range)//3 if price_range else 0}. ' +
                f'Shop short description: {short_desc}. \n\n' +
                'The following are the signature dishes. \n' +
                signature_dishes + '\n' +
                'The following are the sampled reviews. \n' +
                review_contents
            )
            
            document = Document(
                page_content=content,
                metadata={"source": source, "id": doc_id},
                id=str(doc_id),
            )
            doc_id += 1
            documents.append(document)

    # Initialize Embeddings
    credentials = Credentials(url=url, api_key=api_key) if api_key else Credentials(url=url) # Should work if token is managed or env var
    
    # Note: Using API Key if provided, else rely on env vars
    
    embeddings = WatsonxEmbeddings(
        model_id='ibm/slate-30m-english-rtrvr-v2',
        url=url,
        project_id=project_id,
        params={} # Add params if needed
    )

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name='restaurant_collection'
    )
    
    # Create Retriever
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    return retriever

class RAG_Retriever(BaseTool):
    name: str = "RAG Retriever"
    description: str = "Coarse top 10 recommendations based on user profile"
    args_schema: Type[BaseModel] = RetrieverToolInput
    retriever: object = None # To be injected

    def _run(self, user_profile: str) -> str:
        if not self.retriever:
            return "Retriever not initialized."
        try:
            res = self.retriever.invoke(user_profile)
            recommendation_dict = {}
            for i in range(len(res)):
                recommendation_dict[f'Recommendation {i+1}'] = res[i].page_content
            return str(recommendation_dict)
        except Exception as e:
            return f"Error using RAG tool: {str(e)}"

def restaurant_image_analysis(model, img_urls):
    """
    Analyze the images in users' reviews using Llama Vision.
    """
    results = {}
    for i, image_input in enumerate(img_urls):
        curr = []
        if len(image_input) == 0:
            curr.append('No images are included.')
        
        for image_path in image_input:
            try:
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image_bytes = BytesIO(response.content)
                encoded_image = base64.b64encode(image_bytes.read()).decode("utf-8")
                
                prompt = """
                You are a specialist in interpreting food and dining photography.
                You carefully study visual elements to uncover what words often leave out: the vibrancy of presentation, the elegance of plating, the portion size, and even the mood of the dining environment. 
                You understand that users take photos for a reason: sometimes to remember a favorite dish, sometimes to share a beautiful dining setting, and sometimes to celebrate with friends.
                Please give your analysis on the image in one short sentence.
                """
                
                response = model.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                            ],
                        }
                    ]
                )
                
                response_output = response['choices'][0]['message']['content']
                if 'Error' in response_output or 'error' in response_output:
                    continue
                else:
                    curr.append(response_output)
            except Exception as e:
                print(f"Error analyzing image {image_path}: {e}")
                continue
                
        results[f'restaurant visit {i}'] = curr

    return results
