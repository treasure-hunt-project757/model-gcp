from typing import List
from src.models.schemas import DetectableObject
import requests
import os
from dotenv import load_dotenv

# load_dotenv()
# api_url = os.getenv("FETCH_OBJECTS_API")
api_url = os.environ["FETCH_OBJECTS_API"]


def fetch_objects_from_server() -> List[DetectableObject]:
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            objects = response.json()
            return [DetectableObject(**item) for item in objects]

        else:
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error fetching objects from API:", e)
        return None
