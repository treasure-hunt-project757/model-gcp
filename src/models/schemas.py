from pydantic import BaseModel
from typing import List


class DetectableObject(BaseModel):
    objectID: int
    name: str
    locationID: int
    objectImgsUrls: List[str]
