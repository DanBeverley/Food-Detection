import json
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CUSTOM_DB_PATH = "data/databases/custom_density_db.json"
USDA_API_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Global In-Memory Database
_custom_density_db = None
_db_load_attempted = False


def _load_custom_db():
    """
    Loads the custom nutritional database from JSON file into memory.
    This database contains pre-curated density and calorie values for the 108 classes.
    """
    global _custom_density_db, _db_load_attempted
    
    if _db_load_attempted:
        return _custom_density_db

    _db_load_attempted = True
    
    if os.path.exists(CUSTOM_DB_PATH):
        try:
            with open(CUSTOM_DB_PATH, 'r') as f:
                _custom_density_db = json.load(f)
                
            # Remove metadata keys
            if "_comment" in _custom_density_db:
                del _custom_density_db["_comment"]
                
            logger.info(f"Successfully loaded offline nutritional database from {CUSTOM_DB_PATH} ({len(_custom_density_db)} entries)")
            return _custom_density_db
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading custom nutritional database {CUSTOM_DB_PATH}: {e}. Proceeding with empty DB.")
            _custom_density_db = {}
            return _custom_density_db
    else:
        logger.warning(f"Custom nutritional database not found at {CUSTOM_DB_PATH}. Lookups will fail.")
        _custom_density_db = {}
        return _custom_density_db


def lookup_nutritional_info(food_item_name: str, api_key: str = None) -> dict | None:
    """
    Lookup food nutritional info (density g/cm³, calories kcal/100g).

    STRATEGY:
    1. Direct Offline DB Lookup (Primary, O(1), Deterministic)
       - Checks 'custom_density_db.json' for the exact 'food_item_name' (normalized).
       - This covers the fixed 108 classes used by the model.
       
    2. Fallback: USDA API (Secondary, O(n), Fuzzy, Slow)
       - Only triggered if 'api_key' is provided AND item is missing from DB.
       - Not recommended for production due to latency and inconsistency.

    Args:
        food_item_name: Name of the food class (e.g., "Apple", "Fried_Rice").
        api_key: Optional API key for USDA API fallback.

    Returns:
        {'density': float, 'calories_kcal_per_100g': float} or None.
    """
    global _custom_density_db
    
    # 1. Initialize DB if needed
    if _custom_density_db is None:
        _load_custom_db()

    # 2. Normalize Key (Match the generation script logic)
    lookup_key = food_item_name.lower().replace(' ', '_')
    
    # 3. Check Offline Database
    if _custom_density_db and lookup_key in _custom_density_db:
        entry = _custom_density_db[lookup_key]
        
        # Validate entry has required fields
        density = entry.get('density')
        calories = entry.get('calories_kcal_per_100g')
        
        if density is not None and calories is not None:
            logger.info(f"DB Hit: '{food_item_name}' -> Density: {density} g/cm³, Calories: {calories} kcal/100g")
            return {
                'density': float(density),
                'calories_kcal_per_100g': float(calories),
                'protein_g_per_100g': float(entry.get('protein_g_per_100g', 0)),
                'carbs_g_per_100g': float(entry.get('carbs_g_per_100g', 0)),
                'fat_g_per_100g': float(entry.get('fat_g_per_100g', 0))
            }
        else:
            logger.warning(f"DB Entry Incomplete for '{food_item_name}': {entry}")
            # Could fall through to API if partial data, but typically we want the DB to be the source of truth.

    # 4. Fallback: USDA API (Only if configured)
    if api_key:
        logger.info(f"DB Miss for '{food_item_name}'. Attempting USDA API fallback...")
        return _lookup_usda_api_fallback(food_item_name, api_key)
    
    logger.warning(f"Nutritional info not found for '{food_item_name}' in offline DB and no API key provided.")
    return None


def _lookup_usda_api_fallback(query: str, api_key: str) -> dict | None:
    """
    Legacy fallback to USDA API. Kept for edge cases but de-prioritized.
    """
    try:
        url = f"{USDA_API_BASE_URL}?query={requests.utils.quote(query)}&api_key={api_key}&dataType=Foundation,SR%20Legacy"
        response = requests.get(url, timeout=5) # Short timeout
        response.raise_for_status()
        data = response.json()
        
        if not data.get('foods'):
            return None
            
        # Naive first match strategy for fallback
        food = data['foods'][0]
        nutrients = food.get('foodNutrients', [])
        
        density = None
        calories = None
        
        for n in nutrients:
            name = n.get('nutrientName', '').lower()
            # Approximate calorie matching
            if 'energy' in name and 'kcal' in n.get('unitName', '').lower():
                calories = n.get('value')
                
        # USDA API rarely provides density directly, so this often returns partial data
        if calories is not None:
            # density defaults to 0.85 if not found in fallback
            return {'density': 0.85, 'calories_kcal_per_100g': float(calories)}
            
    except Exception as e:
        logger.error(f"USDA API Fallback failed: {e}")
    
    return None 
