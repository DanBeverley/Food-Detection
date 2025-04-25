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
    """Loads the custom density database from JSON file into memory."""
    global _custom_density_db, _custom_db_load_attempted
    if _custom_db_load_attempted:
        return _custom_density_db

    _custom_db_load_attempted = True
    if os.path.exists(CUSTOM_DB_PATH):
        try:
            with open(CUSTOM_DB_PATH, 'r') as f:
                _custom_density_db = json.load(f)
                print(f"Successfully loaded custom density database from {CUSTOM_DB_PATH}")
                # Remove comment entry if it exists
                _custom_density_db.pop("_comment", None)
                return _custom_density_db
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error reading custom density database {CUSTOM_DB_PATH}: {e}. Proceeding without it.")
            _custom_density_db = {} # Ensure it's an empty dict on error
            return _custom_density_db
    else:
        print(f"Warning: Custom density database not found at {CUSTOM_DB_PATH}. Proceeding without it.")
        _custom_density_db = {} # Ensure it's an empty dict if not found
        return _custom_density_db

def lookup_density(food_item: str, api_key: str = None) -> float | None:
    """
    Lookup food density (g/cm³) using cache, custom DB, or USDA API.

    Lookup order:
    1. USDA Cache
    2. Custom Local Database (custom_density_db.json)
    3. USDA API (requires API key)

    Args:
        food_item: Name of the food item (should ideally match classification output
                   and keys in custom_density_db.json).
        api_key: Optional API key for USDA API.

    Returns:
        Density in g/cm³ or None if not found in any source.
    """
    global _custom_density_db
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Use a consistent key format (lowercase, underscore)
    lookup_key = food_item.lower().replace(' ', '_')

    # --- 1. Check Cache ---
    cache_filename = "".join(c for c in lookup_key if c.isalnum() or c == '_') + '.json'
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_filepath):
        try:
            with open(cache_filepath, 'r') as f:
                cached_data = json.load(f)
                density = cached_data.get('density')
                # Check if density is explicitly null (meaning API lookup failed before)
                if 'density' in cached_data: # Distinguish between key missing and key having null value
                     if density is not None:
                        print(f"Cache hit for {food_item}: {density} g/cm³")
                        return float(density)
                     else:
                         print(f"Cache hit (negative) for {food_item}. Density not found.")
                         return None # Explicitly not found
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error reading cache file {cache_filepath}: {e}")
        # If cache exists but density is missing/null, or read error, proceed to next steps

    # --- 2. Check Custom DB ---
    # Load custom DB if not already loaded
    if _custom_density_db is None:
        _load_custom_db()

    if _custom_density_db and lookup_key in _custom_density_db:
        density = _custom_density_db[lookup_key]
        if density is not None:
             print(f"Custom DB hit for {food_item} ({lookup_key}): {density} g/cm³")
             
             return float(density)

    # --- 3. Call USDA API (if key provided and not found above) ---
    if not api_key:
        print(f"Density for {food_item} not found in cache or custom DB. No API key provided.")
        return None # Not found in cache/custom DB, and no API key to check further

    print(f"Cache/Custom DB miss for {food_item}. Calling USDA API.")
    url = f"{USDA_API_BASE_URL}?query={requests.utils.quote(food_item)}&api_key={api_key}&dataType=Foundation,SR%20Legacy" # Broaden search types
    density = None
    api_error = False

    try:
        response = requests.get(url, timeout=15) # Slightly longer timeout
        response.raise_for_status()
        data = response.json()
        foods = data.get('foods', [])

        if foods:
            # Prioritize finding specific gravity (Nutrient ID 271 or 1137)
            # Check multiple foods if necessary, prioritizing exact matches later if needed
            for food in foods:
                # if food.get('description','').lower() == food_item.lower(): # Example simple match
                for nutrient in food.get('foodNutrients', []):
                    # Specific Gravity IDs might vary. Check common ones.
                    if nutrient.get('nutrient', {}).get('id') in [271, 1137] or \
                       'specific gravity' in nutrient.get('nutrient', {}).get('name', '').lower():
                        value = nutrient.get('value')
                        if value is not None:
                            density = float(value)
                            print(f"USDA API hit for {food_item}: Found density {density} g/cm³ in food '{food.get('description')}'")
                            break # Found density, stop searching this food's nutrients
                if density is not None:
                    break # Found density in one of the foods, stop searching foods
            # else: # If loop finished without break
                # print(f"Specific Gravity/Density nutrient not found in API response for '{food.get('description')}'.")

        if density is None:
             print(f"Density information not found for {food_item} in USDA API response.")

    except requests.exceptions.Timeout:
        print(f"Error: API request timed out for {food_item}.")
        api_error = True
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed for {food_item}: {e}")
        api_error = True
    except Exception as e: # Catch potential JSON parsing or other errors
        print(f"Error: Unexpected error processing API response for {food_item}: {e}")
        api_error = True

    # --- 4. Cache Result (even if not found) & Return ---
    if not api_error:
        # Cache the result from the API call
        try:
            with open(cache_filepath, 'w') as f:
                json.dump({'food_item': food_item, 'density': density, 'timestamp': time.time()}, f, indent=2)
                action = "cached" if density is not None else "cached negative result for"
                print(f"Successfully {action} {food_item}")
        except IOError as e:
            print(f"Warning: Error writing cache file {cache_filepath}: {e}")

        # --- 5. Update Custom DB if API found density and item is not already present ---
        if density is not None and _custom_density_db is not None:
            if lookup_key not in _custom_density_db:
                print(f"Adding API result for '{lookup_key}' to custom density database.")
                _custom_density_db[lookup_key] = density
                # Write the updated custom DB back to the file
                try:
                    with open(CUSTOM_DB_PATH, 'w') as f:
                        # Add the comment back for clarity when manually viewing
                        db_to_write = {"_comment": "Populate this file with food items (matching classification output) and their densities in g/cm3. Example: \"apple_fuji_raw\": 0.75"}
                        db_to_write.update(_custom_density_db)
                        json.dump(db_to_write, f, indent=2)
                except IOError as e:
                    print(f"Warning: Error writing updated custom density database {CUSTOM_DB_PATH}: {e}")
            # else: Item already exists in custom DB, do not overwrite with API result.

    return density # Returns the found density (float) or None


if __name__ == '__main__':
    API_KEY = os.environ.get("USDA_API_KEY")
    if not API_KEY:
         print("Warning: USDA_API_KEY environment variable not set. API lookups will be skipped.")

    test_foods = ["apple", "cooked white rice", "butter", "water", "nonexistent_food_xyz", "apple_fuji_raw", "banana_raw"]

    # Load custom DB once before testing
    _load_custom_db()

    for food in test_foods:
        print(f"\n--- Looking up: {food} ---")
        density_value = lookup_density(food, api_key=API_KEY)
        if density_value is not None:
            print(f"===> Final Density for {food}: {density_value} g/cm³")
        else:
            print(f"===> Final Density for {food}: Not found")

    # Example of adding to custom DB (in a real app, this might be separate)
    # _custom_density_db['orange_navel_raw'] = 0.96
    # print("\n--- Looking up orange (after potential custom add) ---")
    # density_value = lookup_density("orange navel raw") # No API key needed if in custom DB
    # print(f"===> Final Density for orange navel raw: {density_value}")
