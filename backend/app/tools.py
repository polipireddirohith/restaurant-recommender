import os
import json
import requests
import base64
from io import BytesIO
from typing import List, Dict, Type

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crewai.tools import BaseTool, tool
from app.models import RetrieverToolInput  # Import from local package
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def setup_rag(api_key=None):
    """
    Initializes the RAG retriever tool using Google Gemini Embeddings.
    """
    if not api_key:
        raise ValueError("Google Gemini API Key must be provided.")

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
    print(f"Initializing Gemini Embeddings...")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    candidates = ["models/text-embedding-004", "models/embedding-001"]
    try:
        available = [m.name for m in genai.list_models() if "embedContent" in m.supported_generation_methods]
        print(f"Discovered embedding models: {available}")
        import sys
        sys.stdout.flush()
        for m in available:
            if m not in candidates:
                candidates.append(m)
    except Exception as e:
        print(f"Note: Could not list embedding models: {e}")
        import sys
        sys.stdout.flush()

    embeddings = None
    for emb_model in candidates:
        try:
            print(f"Trying embedding model: {emb_model}...")
            import sys
            sys.stdout.flush()
            embeddings = GoogleGenerativeAIEmbeddings(
                model=emb_model,
                google_api_key=api_key
            )
            # Test it
            embeddings.embed_query("test")
            print(f"Selected embedding model: {emb_model}")
            sys.stdout.flush()
            break
        except Exception as e:
            print(f"Embedding model {emb_model} failed: {e}")
            embeddings = None
            continue

    if not embeddings:
        print("FAILED to initialize any embedding model.")
        return None

    print(f"Embedding {len(documents)} documents into Chroma...")
    try:
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name='restaurant_collection'
        )
        print("Vector database created successfully.")
    except Exception as e:
        print(f"FAILED to create vector database: {e}")
        raise e
    
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
    Analyze the images in users' reviews using Gemini Vision.
    """
    results = {}
    for i, image_input in enumerate(img_urls):
        curr = []
        if len(image_input) == 0:
            curr.append('No images are included.')
        
        for image_path in image_input:
            try:
                # Gemini Pro Vision handling via model.invoke would be needed here
                # For simplicity in this demo, we'll keep the structure but skip complex vision logic 
                # unless specifically requested, as it requires specific prompt formatting for Gemini
                curr.append("Image analysis placeholder for Gemini Vision.")
            except Exception as e:
                print(f"Error analyzing image {image_path}: {e}")
                continue
                
        results[f'restaurant visit {i}'] = curr

    return results
