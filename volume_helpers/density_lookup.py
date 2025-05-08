import requests
import json
import os
import time

CACHE_DIR = "data/databases/usda_cache"
CUSTOM_DB_PATH = "data/databases/custom_density_db.json"
USDA_API_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

_custom_density_db = None
_custom_db_load_attempted = False

def _load_custom_db():
    """Loads the custom nutritional database from JSON file into memory."""
    global _custom_density_db, _custom_db_load_attempted
    if _custom_db_load_attempted:
        return _custom_density_db

    _custom_db_load_attempted = True
    if os.path.exists(CUSTOM_DB_PATH):
        try:
            with open(CUSTOM_DB_PATH, 'r') as f:
                _custom_density_db = json.load(f)
                print(f"Successfully loaded custom nutritional database from {CUSTOM_DB_PATH}")
                _custom_density_db.pop("_comment", None)
                return _custom_density_db
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error reading custom nutritional database {CUSTOM_DB_PATH}: {e}. Proceeding without it.")
            _custom_density_db = {} 
            return _custom_density_db
    else:
        print(f"Warning: Custom nutritional database not found at {CUSTOM_DB_PATH}. Proceeding without it.")
        _custom_density_db = {} 
        return _custom_density_db

def lookup_nutritional_info(food_item: str, api_key: str = None) -> dict | None:
    """
    Lookup food nutritional info (density g/cm³, calories kcal/100g) using cache, custom DB, or USDA API.

    Lookup order:
    1. USDA Cache
    2. Custom Local Database (custom_density_db.json)
    3. USDA API (requires API key)

    Args:
        food_item: Name of the food item.
        api_key: Optional API key for USDA API.

    Returns:
        A dictionary {'density': float | None, 'calories_kcal_per_100g': float | None} 
        or None if the food item itself is definitively not found after all checks.
    """
    global _custom_density_db
    os.makedirs(CACHE_DIR, exist_ok=True)

    lookup_key = food_item.lower().replace(' ', '_')
    cache_filename = "".join(c for c in lookup_key if c.isalnum() or c == '_') + '.json'
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    result_data = {'density': None, 'calories_kcal_per_100g': None}

    if os.path.exists(cache_filepath):
        try:
            with open(cache_filepath, 'r') as f:
                cached_data = json.load(f)
                if cached_data.get('status') == 'not_found':
                    print(f"Cache hit (negative - item not found) for {food_item}.")
                    return None 

                if 'density' in cached_data:
                    result_data['density'] = cached_data['density']
                if 'calories_kcal_per_100g' in cached_data:
                    result_data['calories_kcal_per_100g'] = cached_data['calories_kcal_per_100g']
                
                if result_data['density'] is not None or result_data['calories_kcal_per_100g'] is not None:
                    print(f"Cache hit for {food_item}: {result_data}")
                    if result_data['density'] is not None and result_data['calories_kcal_per_100g'] is not None:
                         return result_data
                elif cached_data.get('density', -1) is None and cached_data.get('calories_kcal_per_100g', -1) is None and 'status' not in cached_data:
                    print(f"Cache hit (explicit nulls) for {food_item}. Information not found previously.")
                    pass 
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error reading cache file {cache_filepath}: {e}")

    if _custom_density_db is None:
        _load_custom_db()

    if _custom_density_db and lookup_key in _custom_density_db:
        db_entry = _custom_density_db[lookup_key]
        if isinstance(db_entry, dict):
            if result_data['density'] is None and 'density' in db_entry:
                result_data['density'] = db_entry['density']
            if result_data['calories_kcal_per_100g'] is None and 'calories_kcal_per_100g' in db_entry:
                result_data['calories_kcal_per_100g'] = db_entry['calories_kcal_per_100g']
            
            if db_entry.get('density') is not None or db_entry.get('calories_kcal_per_100g') is not None:
                print(f"Custom DB hit for {food_item} ({lookup_key}): {db_entry}")
                if result_data['density'] is not None and result_data['calories_kcal_per_100g'] is not None:
                     return result_data
        elif isinstance(db_entry, (float, int)) and result_data['density'] is None:
            result_data['density'] = float(db_entry)
            print(f"Custom DB (legacy format) hit for {food_item} ({lookup_key}): density {result_data['density']} g/cm³")

    if not api_key and (result_data['density'] is None or result_data['calories_kcal_per_100g'] is None):
        print(f"Nutritional info for {food_item} not fully found in cache/custom DB. No API key provided.")
        return result_data if result_data['density'] is not None or result_data['calories_kcal_per_100g'] is not None else None

    if api_key and (result_data['density'] is None or result_data['calories_kcal_per_100g'] is None):
        print(f"Cache/Custom DB miss or incomplete for {food_item}. Calling USDA API.")
        url = f"{USDA_API_BASE_URL}?query={requests.utils.quote(food_item)}&api_key={api_key}&dataType=Foundation,SR%20Legacy,Survey%20(FNDDS),Branded"
        api_error = False
        api_found_density = None
        api_found_calories = None

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            foods = data.get('foods', [])

            if not foods:
                print(f"No food items found in USDA API response for query: {food_item}")
                # Cache this negative result for the food item itself
                with open(cache_filepath, 'w') as f:
                    json.dump({'food_item': food_item, 'status': 'not_found', 'timestamp': time.time()}, f, indent=2)
                return None # Food item not found by API

            # Iterate through found foods to find relevant nutrients
            # Log details for the first few food items from the API response for debugging
            MAX_FOODS_TO_DEBUG_LOG = 3
            for i, food_data_item_debug in enumerate(foods[:MAX_FOODS_TO_DEBUG_LOG]):
                debug_description = food_data_item_debug.get('description', 'N/A')
                debug_fdcId = food_data_item_debug.get('fdcId', 'N/A')
                print(f"DEBUG USDA API: Checking food item {i+1}/{len(foods)}: '{debug_description}' (FDC ID: {debug_fdcId})")
                debug_nutrients = food_data_item_debug.get('foodNutrients', [])
                if not debug_nutrients:
                    print(f"DEBUG USDA API:   No nutrients listed for '{debug_description}'.")
                else:
                    for nutrient_debug in debug_nutrients[:15]: # Log first 15 nutrients to avoid excessive output
                        nutrient_info_name = nutrient_debug.get('nutrientName', '').lower()
                        nutrient_info_id = nutrient_debug.get('nutrientId', 'N/A')
                        nutrient_info_unit = nutrient_debug.get('unitName', '').upper() # FDC API often uses nutrient.unitName
                        if not nutrient_info_unit or nutrient_info_unit == 'N/A': # Fallback for SR Legacy style if needed
                            nutrient_info_unit = nutrient_debug.get('nutrient', {}).get('unitName', 'N/A') 
                        nutrient_value = nutrient_debug.get('value', nutrient_debug.get('amount', 'N/A'))
                        print(f"DEBUG USDA API:   Nutrient: {nutrient_info_name} (ID: {nutrient_info_id}), Unit: {nutrient_info_unit}, Value: {nutrient_value}")
                    if len(debug_nutrients) > 15:
                        print(f"DEBUG USDA API:   ... and {len(debug_nutrients) - 15} more nutrients not logged for brevity.")

            for food_data_item in foods: # Renamed 'food' to 'food_data_item' to avoid conflict
                # Only process if we still need density or calories
                if api_found_density is not None and api_found_calories is not None:
                    break # Already found both

                food_description = food_data_item.get('description', 'N/A')
                nutrients = food_data_item.get('foodNutrients', [])
                
                # Try to find density if still needed
                if api_found_density is None:
                    for nutrient in nutrients:
                        # Try direct access first (common in newer FDC API responses)
                        parsed_nutrient_name = nutrient.get('nutrientName', '').lower()
                        parsed_nutrient_id = nutrient.get('nutrientId')
                        
                        # Fallback to nested 'nutrient' object (common in SR Legacy or older formats)
                        if not parsed_nutrient_name and parsed_nutrient_id is None: # Check ID is None, as 0 is a valid ID
                            nutrient_details_obj = nutrient.get('nutrient', {})
                            parsed_nutrient_name = nutrient_details_obj.get('name', '').lower()
                            parsed_nutrient_id = nutrient_details_obj.get('id')

                        if parsed_nutrient_id in [271, 1137] or 'specific gravity' in parsed_nutrient_name:
                            value = nutrient.get('value', nutrient.get('amount')) # Some API versions use 'amount'
                            if value is not None:
                                try:
                                    api_found_density = float(value)
                                    print(f"USDA API: Found density {api_found_density} g/cm³ for '{food_description}' (query: {food_item})")
                                    break # Found density for this food item
                                except ValueError:
                                    print(f"Warning: Could not convert density value '{value}' to float for '{food_description}'.")
                
                # Try to find calories if still needed
                if api_found_calories is None:
                    for nutrient in nutrients:
                        # Try direct access first
                        parsed_nutrient_name = nutrient.get('nutrientName', '').lower()
                        parsed_nutrient_id = nutrient.get('nutrientId')
                        parsed_unit_name = nutrient.get('unitName', '').upper()

                        # Fallback to nested 'nutrient' object
                        if not parsed_nutrient_name and parsed_nutrient_id is None: # If primary name/id failed, try nested for all
                            nutrient_details_obj = nutrient.get('nutrient', {})
                            parsed_nutrient_name = nutrient_details_obj.get('name', '').lower()
                            parsed_nutrient_id = nutrient_details_obj.get('id')
                            # If unit name was also not found directly, try nested for it too
                            if not parsed_unit_name: # Covers empty string and None
                                 parsed_unit_name = nutrient_details_obj.get('unitName', '').upper()
                        
                        # Check for Nutrient ID 208 (Energy in kcal) or name 'Energy' and unit 'KCAL'
                        # Nutrient ID 1008 is also sometimes used for Energy in kcal in SR Legacy
                        if (parsed_nutrient_id in [208, 1008]) or \
                           ('energy' in parsed_nutrient_name and parsed_unit_name == 'KCAL'):
                            value = nutrient.get('value', nutrient.get('amount'))
                            if value is not None:
                                try:
                                    api_found_calories = float(value) # Calories are per 100g by default in USDA
                                    print(f"USDA API: Found calories {api_found_calories} kcal/100g for '{food_description}' (query: {food_item})")
                                    break # Found calories for this food item
                                except ValueError:
                                    print(f"Warning: Could not convert calorie value '{value}' to float for '{food_description}'.")
                
                # If we found both for this food_data_item, no need to check other food_data_items from API results
                if api_found_density is not None and api_found_calories is not None:
                    break
            
            if api_found_density is None and result_data['density'] is None: # Only log if not found in API and not from cache/custom
                 print(f"Density information not found for {food_item} in USDA API response after checking all foods.")
            if api_found_calories is None and result_data['calories_kcal_per_100g'] is None:
                 print(f"Calorie (kcal) information not found for {food_item} in USDA API response after checking all foods.")

        except requests.exceptions.Timeout:
            print(f"Error: API request timed out for {food_item}.")
            api_error = True
        except requests.exceptions.RequestException as e:
            print(f"Error: API request failed for {food_item}: {e}")
            api_error = True
        except Exception as e: 
            print(f"Error: Unexpected error processing API response for {food_item}: {e}")
            api_error = True

    if api_key or (_custom_density_db and lookup_key in _custom_density_db) :
        if result_data['density'] is None and api_found_density is not None:
            result_data['density'] = api_found_density
        if result_data['calories_kcal_per_100g'] is None and api_found_calories is not None:
            result_data['calories_kcal_per_100g'] = api_found_calories

        if not api_error:
            try:
                cache_entry = {
                    'food_item': food_item,
                    'density': float(result_data['density']) if result_data['density'] is not None else None,
                    'calories_kcal_per_100g': float(result_data['calories_kcal_per_100g']) if result_data['calories_kcal_per_100g'] is not None else None,
                    'timestamp': time.time()
                }
                with open(cache_filepath, 'w') as f:
                    json.dump(cache_entry, f, indent=2)
                action = "updated cache for"
                print(f"Successfully {action} {food_item} with: {cache_entry}")
            except IOError as e:
                print(f"Warning: Error writing cache file {cache_filepath}: {e}")

    if result_data['density'] is not None or result_data['calories_kcal_per_100g'] is not None:
        return result_data
    return None 


if __name__ == '__main__':
    API_KEY = os.environ.get("USDA_API_KEY")
    if not API_KEY:
         print("Warning: USDA_API_KEY environment variable not set. API lookups will be skipped.")

    test_foods = ["apple", "cooked white rice", "butter", "water", "nonexistent_food_xyz", "apple_fuji_raw", "banana_raw"]

    _load_custom_db() 

    for food in test_foods:
        print(f"\n--- Looking up: {food} ---")
        nutritional_info = lookup_nutritional_info(food, api_key=API_KEY)
        if nutritional_info:
            print(f"===> Final Nutritional Info for {food}: {nutritional_info}")
        else:
            print(f"===> Final Nutritional Info for {food}: Not found")

    # Example: How the custom DB should look
    # _custom_density_db['orange_navel_raw'] = {'density': 0.96, 'calories_kcal_per_100g': 47}
    # print("\n--- Looking up orange (after potential custom add) ---")
    # info = lookup_nutritional_info("orange navel raw", api_key=API_KEY)
    # print(f"===> Final Nutritional Info for orange navel raw: {info}")
