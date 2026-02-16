import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_default_history():
    try:
        review_path = os.path.join(DATA_DIR, 'review-user.csv')
        item_path = os.path.join(DATA_DIR, 'restaurant-item.csv')

        if not os.path.exists(review_path) or not os.path.exists(item_path):
            print("Data files not found.")
            return []

        target_user_reviews = pd.read_csv(review_path)
        restaurant_info = pd.read_csv(item_path)

        # Mapping functions
        price_interval_map = {np.nan:0, '$':1, '$$ - $$$':2, '$$$$':3}
        
        def rating_map(rating):
            return rating/10.0
        
        def itemId_map(itemId):
            target_restaurant = restaurant_info[restaurant_info['itemId']==itemId]
            if len(target_restaurant) == 0:
                return 'This restaurant has no information.'
            
            short_summary = f"This is a restaurant of type {target_restaurant['type'].iloc[0]} with price interval {target_restaurant['priceInterval'].iloc[0]}. It has average rating {target_restaurant['rating'].iloc[0]/10.0}."
            return short_summary

        # Process
        # 1. Image URLs (Skipping for now as they match by index or ID in notebook logic but here simpler)
        # 2. Select columns
        target_user_reviews = target_user_reviews[['itemId','title','text','rating']]

        # 3. Map
        # Note: In the notebook, restaurant_info['priceInterval'] is modified globally. 
        # We need to act on copy or just allow it.
        # But we need to be careful about setting with copy warning.
        restaurant_cop = restaurant_info.copy()
        restaurant_cop['priceInterval'] = restaurant_cop['priceInterval'].map(price_interval_map)

        # itemId_map uses restaurant_info (the original DF in closure). 
        # But we modified the copy. We should use the modified copy inside the function?
        # The notebook modifies restaurant_info in place.
        restaurant_info['priceInterval'] = restaurant_info['priceInterval'].map(price_interval_map) 

        target_user_reviews['rating'] = target_user_reviews['rating'].map(rating_map)
        target_user_reviews['itemId'] = target_user_reviews['itemId'].map(itemId_map)

        # 4. Rename
        target_user_reviews = target_user_reviews.rename(columns={'itemId': 'restaurant info', 'title': 'review title', 'text': 'review text'})

        # 5. To Dict
        visit_history = target_user_reviews.to_dict(orient="records")
        return visit_history
    except Exception as e:
        print(f"Error loading default history: {e}")
        return []
