import json
import os
import time
import requests
import argparse
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USDA_API_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
CUSTOM_DB_PATH = "data/databases/custom_density_db.json"
LABEL_MAP_PATH = "data/classification/label_map.json"

# --- DEFAULTS (From Refactored Logic) ---
CATEGORIES = {
    "Fruit": {"density": 0.85, "calories": 60},
    "Vegetable": {"density": 0.50, "calories": 35}, 
    "Meat": {"density": 1.05, "calories": 250},      
    "Fried": {"density": 0.70, "calories": 300},     
    "Carb_Bread": {"density": 0.40, "calories": 280},
    "Carb_Rice": {"density": 0.85, "calories": 130},  
    "Dairy": {"density": 1.02, "calories": 150},
    "Soup": {"density": 1.00, "calories": 50},
    "Sweet": {"density": 0.65, "calories": 350},
    "Nut": {"density": 0.60, "calories": 600},       
    "Seafood": {"density": 1.02, "calories": 120},
    "Default": {"density": 0.85, "calories": 150}
}

CLASS_CATEGORY_MAP = {
    # Fruit
    "Apple": "Fruit", "Applesauce": "Fruit", "Avocado": "Fruit", "Banana": "Fruit",
    "Blueberries": "Fruit", "Blueberry(bowl)": "Fruit", "Grapes": "Fruit", "Kiwi": "Fruit",
    "Mango": "Fruit", "Melons": "Fruit", "Papaya": "Fruit", "Peach": "Fruit",
    "Pineapple": "Fruit", "Strawberry": "Fruit", "Watermelon": "Fruit", 

    # Vegetable
    "Asparagus": "Vegetable", "Beans": "Vegetable", "Bell Pepper": "Vegetable",
    "Broccoli": "Vegetable", "Cabbage": "Vegetable", "Carrot": "Vegetable",
    "Cauliflower": "Vegetable", "Coleslaw": "Vegetable", "Corn(bowl)": "Vegetable",
    "Corn_Stick": "Vegetable", "Cucumber": "Vegetable", "Edamame": "Vegetable",
    "Green_Bean": "Vegetable", "Green_beans": "Vegetable", "Guacamole": "Vegetable",
    "Onion": "Vegetable", "Sweet Potato": "Vegetable", "Tomato": "Vegetable",
    "Tomato_slice": "Vegetable",

    # Meat
    "Bacon": "Meat", "Burger": "Meat", "Chicken_breast": "Meat", "Chicken_tender": "Meat",
    "Chicken_thighs": "Meat", "Chicken_wings": "Meat", "Frankfurter_Sandwich": "Meat",
    "Meatloaf": "Meat", "Pork_Chop": "Meat", "Pork_Rib": "Meat", "Sausages": "Meat",
    "Steak": "Meat", "Stew_Beef": "Soup", "Whole_Chicken": "Meat",

    # Seafood
    "Breaded_Fish": "Fried", "Crab": "Seafood", "Crab cake": "Fried", "Lobster": "Seafood",
    "Salmon_Grill": "Seafood", "Shrimp": "Seafood", "Sushi": "Carb_Rice", "Tuna_salad": "Seafood",

    # Fried / Fast Food
    "Chicken_Nugget": "Fried", "Corn dog": "Fried", "French_Fry": "Fried",
    "Fried mushrooms": "Fried", "Hashbrown": "Fried", "Nachos": "Fried",
    "Tortilla_Chips": "Fried",

    # Carb - Bread/Baked
    "Bagel": "Carb_Bread", "Biscuits": "Carb_Bread", "Cinnamon_Roll": "Sweet",
    "Croissant": "Carb_Bread", "Doughnut": "Sweet", "French_toast": "Carb_Bread",
    "Muffin": "Carb_Bread", "Pancake": "Carb_Bread", "Pizza": "Carb_Bread",
    "Quesadilla": "Carb_Bread", "Quickbread": "Carb_Bread", "Sandwich": "Carb_Bread",
    "Taco": "Carb_Bread", "Tortilla": "Carb_Bread", "Waffle": "Carb_Bread",
    "Yeast_bread": "Carb_Bread",

    # Carb - Rice/Pasta
    "Fried_rice": "Carb_Rice", "Gyoza": "Carb_Rice", "Lasagna": "Carb_Rice",
    "Mac": "Carb_Rice", "Pasta_mixed_dishes": "Carb_Rice", "Rice": "Carb_Rice",

    # Dairy / Egg
    "Cottage cheese": "Dairy", "Egg": "Dairy", "Fried_egg": "Dairy",
    "Ice_cream": "Sweet", "Omelet": "Dairy", "Yogurt": "Dairy",
    "Pudding": "Sweet",

    # Nuts / Seeds
    "Almond(bowl)": "Nut", "Almonds": "Nut", "Peanuts": "Nut", 
    "Peanut butter and jelly sandwiches": "Carb_Bread",

    # Sweets
    "Brownies": "Sweet", "Cake": "Sweet", "Cereal bars": "Sweet",
    "Chocolate": "Sweet", "Cookies": "Sweet", "Nutrition bars": "Sweet",
    "Pie": "Sweet",

    # Other
    "Baked_Potato": "Vegetable", "Burrito": "Carb_Rice", "Falafel": "Fried",
    "Mashed_Potato": "Vegetable", "Soup": "Soup", "Tofu": "Vegetable"
}

def generate_default_db(target_path: str):
    """Generates the initial DB from defaults if it doesn't exist."""
    print(f"--- Generating Default Database ---")
    
    if not os.path.exists(LABEL_MAP_PATH):
        logging.error(f"Error: {LABEL_MAP_PATH} not found. Ensure you have prepared the classification dataset first.")
        return

    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)

    db = {
        "_comment": "Generated offline database. Values are category-based estimates or cached from API."
    }

    logging.info(f"Seeding DB for {len(label_map)} classes...")

    for idx, class_name in label_map.items():
        category_key = CLASS_CATEGORY_MAP.get(class_name, "Default")
        vals = CATEGORIES[category_key]
        
        lookup_key = class_name.lower().replace(" ", "_").strip()
        
        db[lookup_key] = {
            "density": vals['density'],
            "calories_kcal_per_100g": vals['calories'],
            "original_name": class_name,
            "category": category_key
        }

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w') as f:
        json.dump(db, f, indent=2)
    
    logging.info(f"Database initialized at {target_path} with {len(db)-1} entries.")


def fetch_usda_data(query: str, api_key: str) -> Dict[str, Any] | None:
    """Queries USDA API."""
    try:
        clean_query = query.replace("_", " ").replace("(bowl)", "").strip()
        
        params = {
            'query': clean_query,
            'api_key': api_key,
            'pageSize': 3
        }
        
        response = requests.get(USDA_API_BASE_URL, params=params, timeout=10)
        
        if response.status_code == 429:
            logging.warning("USDA API Rate limit exceeded. Waiting 60s...")
            time.sleep(60)
            return fetch_usda_data(query, api_key) 
            
        if response.status_code != 200:
             logging.error(f"API Error {response.status_code}: {response.text}")
             
        response.raise_for_status()
        data = response.json()
        
        if not data.get('foods'):
            return None
            
        best_match = None
        for food in data['foods'][:3]:
            result = {
                'calories': None, 'density': None, 
                'protein': None, 'carbs': None, 'fat': None
            }
            nutrients = food.get('foodNutrients', [])
            
            for n in nutrients:
                name = n.get('nutrientName', '').lower()
                unit = n.get('unitName', '').lower()
                val = n.get('value')

                if ('energy' in name and 'kcal' in unit) or n.get('nutrientId') in [208, 1008]:
                    result['calories'] = val
                if n.get('nutrientId') == 203 or ('protein' in name and 'g' in unit):
                    result['protein'] = val
                if n.get('nutrientId') == 205 or ('carbohydrate' in name and 'g' in unit):
                    result['carbs'] = val
                if n.get('nutrientId') == 204 or ('total lipid' in name and 'g' in unit):
                    result['fat'] = val
                if 'specific gravity' in name or n.get('nutrientId') in [271, 1137]:
                    result['density'] = val
            
            if result['calories'] is not None:
                best_match = result
                if result['density'] is not None:
                    break 
        return best_match

    except Exception as e:
        logging.error(f"Error fetching data for '{query}': {e}")
        return None

def update_db_from_usda(db_path: str, api_key: str):
    """Updates existing DB with USDA data."""
    print(f"\n--- Updating Database from USDA API ---")
    
    if not os.path.exists(db_path):
        logging.error(f"Database {db_path} missing in update phase.")
        return

    with open(db_path, 'r') as f:
        db = json.load(f)

    updated_count = 0
    logging.info(f"Processing {len(db)-1} food items...")

    for key, entry in db.items():
        if key.startswith("_"): continue

        # Skip if already fetched (unless forced? We'll assume check source)
        # Uncomment this line if you want to skip already updated items
        # if entry.get('source') == "USDA_API_FETCHED": continue
        
        original_name = entry.get('original_name', key)
        logging.info(f"Querying: {original_name}")
        
        # Rate limit safety
        time.sleep(0.5)
        
        api_result = fetch_usda_data(original_name, api_key)
        
        if api_result:
            updates = {
                'calories_kcal_per_100g': 'calories',
                'protein_g_per_100g': 'protein',
                'carbs_g_per_100g': 'carbs',
                'fat_g_per_100g': 'fat',
                'density': 'density'
            }
            
            changed = False
            for db_field, api_field in updates.items():
                if api_result.get(api_field) is not None:
                    entry[db_field] = api_result[api_field]
                    changed = True
            
            if changed:
                entry['source'] = "USDA_API_FETCHED"
                updated_count += 1
                logging.info(f"  -> Updated {original_name} with API data.")
        else:
            logging.warning(f"  -> No match found for {original_name}")

        # Incremental Save (Every 10 items)
        if (list(db.keys()).index(key) + 1) % 10 == 0:
            logging.info("Saving progress...")
            with open(db_path, 'w') as f:
                json.dump(db, f, indent=2)

    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)
    
    logging.info(f"Update Complete. {updated_count} items updated with real data.")

def main():
    parser = argparse.ArgumentParser(description="Manage Offline Food Density/Nutrition Database")
    parser.add_argument("--api_key", help="USDA API Key for populating/updating data")
    parser.add_argument("--reset", action="store_true", help="Force regenerate defaults (WARNING: Overwrites existing DB)")
    args = parser.parse_args()
    
    # 1. Check if DB exists
    db_exists = os.path.exists(CUSTOM_DB_PATH)
    
    # 2. Generate if missing or forced
    if args.reset or not db_exists:
        generate_default_db(CUSTOM_DB_PATH)
        db_exists = True # assuming success
    
    # 3. Update if API key provided
    if args.api_key:
        if db_exists:
            update_db_from_usda(CUSTOM_DB_PATH, args.api_key)
        else:
            logging.error("Cannot update: Database creation failed.")

if __name__ == "__main__":
    main()
