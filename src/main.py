from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from . import smokeTest
from fastapi import FastAPI, File, HTTPException, UploadFile
from src.models.model import ModelHandler
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from src.models.schemas import DetectableObject
from src.utils.api_utils import fetch_objects_from_server
from typing import List, Optional


app = FastAPI()
model_handler = ModelHandler()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(smokeTest.router, prefix="/smoke-test")


@app.get("/")
def health():
    return {"message": "OK 🍾 "}


@app.get("/get_objects")
async def get_objects():
   try:
       fetch_objects_from_server()
       return {"status": "got em"}
   except Exception as e: 
       raise HTTPException(status_code=500, detail=f"Error getting object : {str(e)}")


@app.post("/retrain")
async def retrain_model():
    try:
        model_handler.retrain()
        return {"status": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {str(e)}")


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    keras_result, keras_max_prob = model_handler.predict(image)
    print("keras_result: ", keras_result, "keras_max_prob: ", keras_max_prob)
    return {"keras_result": keras_result, "keras_max_prob": keras_max_prob}


# @app.get("/gcp_info")
# async def get_info():
#     return {"gcp_info": check_gcs_connection()}


@app.get("/check_objects")
async def check_objects():
    try:
        objects: Optional[List[DetectableObject]] = fetch_objects_from_server()
        if objects is None:
            raise HTTPException(
                status_code=500, detail="Failed to fetch objects from server"
            )
        if len(objects) == 0:
            return {
                "status": "success",
                "message": "No objects found",
                "object_count": 0,
            }
        return {
            "status": "success",
            "object_count": len(objects),
            "objects": [obj.dict() for obj in objects],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/wakeup")
async def wakeup():
    try:
        model, labels = model_handler.get_cached_model()
        if model is not None and labels is not None:
            return JSONResponse(
                status_code=200,
                content={"message": "Cache successfully retrieved", "status": "OK"},
            )
        else:
            return JSONResponse(
                status_code=503,
                content={"message": "Cache not ready", "status": "Service Unavailable"},
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
