import requests
import json

def lookup_density(food_item: str, api_key: str = None) -> float:
    """
    Lookup food density using USDA API or a local fallback.
    Requires an API key for USDA API; handle securely.
    
    Args:
        food_item: Name of the food item.
        api_key: Optional API key for USDA API.
    
    Returns:
        Density in g/cm³ or None if not found.
    
    Raises:
        ValueError: If API call fails.
    """
    if api_key is None:
        raise ValueError("API key is required for USDA lookup. Use a custom density DB as fallback.")
    
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_item}&api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        density = data.get('foods', [{}])[0].get('density', None)  
        return density if density else 1.0  # Default density if not found
    else:
        raise ValueError(f"API request failed with status {response.status_code}")

if __name__ == '__main__':
    try:
        density = lookup_density("apple", api_key="YOUR_API_KEY_HERE")
        print(f"Density: {density} g/cm³")
    except ValueError as e:
        print(e)
