from google.cloud import storage
import os
from google.auth.exceptions import DefaultCredentialsError
import requests
from typing import List

import tensorflow as tf
from src.models.schemas import DetectableObject
import tempfile

storage_client = storage.Client()
BUCKET_NAME = "project-files-storage-clone"


def check_gcs_connection():
    try:
        storage_client = storage.Client()

        buckets = list(storage_client.list_buckets(max_results=1))
        return {
            "status": "success",
            "message": "Successfully connected to Google Cloud Storage",
            "client_methods": dir(storage_client),
        }
    except DefaultCredentialsError:
        return {
            "status": "error",
            "message": "Failed to authenticate. Check your credentials.",
            "client_methods": None,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "client_methods": None,
        }


def download_images(objects: List[DetectableObject]) -> str:
    """
    Downloads images for each DetectableObject using requests and stores them in a temporary directory.

    Args:
    objects (List[DetectableObject]): List of DetectableObject instances containing image URLs.

    Returns:
    str: Path to the temporary directory containing downloaded images.
    """
    temp_dir = tempfile.mkdtemp(prefix="dataset_")

    for obj in objects:
        class_dir = os.path.join(temp_dir, obj.name)
        os.makedirs(class_dir, exist_ok=True)

        for i, img_url in enumerate(obj.objectImgsUrls):
            try:
                response = requests.get(img_url, stream=True)
                if response.status_code == 200:
                    file_extension = (
                        os.path.splitext(img_url.split("/")[-1])[-1] or ".jpg"
                    )
                    local_filename = os.path.join(
                        class_dir, f"{obj.name}_{i}{file_extension}"
                    )

                    with open(local_filename, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print(f"Downloaded {img_url} to {local_filename}")
                else:
                    print(
                        f"Failed to download {img_url}. Status code: {response.status_code}"
                    )

            except Exception as e:
                print(f"Error downloading {img_url}: {str(e)}")

    return temp_dir


def save_model_to_gcs(model, class_indices, model_dir: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_model_path = os.path.join(temp_dir, "model.keras")  ## save to tmp file
        model.save(temp_model_path)

        keras_model_path = f"{model_dir}/model.keras"
        blob = bucket.blob(keras_model_path)
        blob.upload_from_filename(temp_model_path)

        labels_path = f"{model_dir}/labels.txt"
        labels_content = "\n".join(
            f"{class_name}: {index}" for class_name, index in class_indices.items()
        )
        blob = bucket.blob(labels_path)
        blob.upload_from_string(labels_content)

        ############### tflite -- might be deleted ###############

        # try:
        #     converter = tf.lite.TFLiteConverter.from_keras_model(model)
        #     tflite_model = converter.convert()

        #     tflite_model_path = f"{model_dir}/model.tflite"
        #     blob = bucket.blob(tflite_model_path)
        #     blob.upload_from_string(tflite_model)
        #     print(
        #         f"TFLite model saved successfully at gs://{BUCKET_NAME}/{tflite_model_path}"
        #     )
        # except Exception as e:
        #     print(f"Error converting model to TFLite format: {e}")

    return f"gs://{BUCKET_NAME}/{model_dir}"


import logging


def create_model_directory(base_dir: str = "models") -> str:
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    base_name = "model"
    count = 0
    model_dir = base_name

    while True:
        blobs = list(
            bucket.list_blobs(prefix=f"{base_dir}/{model_dir}/", max_results=1)
        )
        if not blobs:
            break
        count += 1
        model_dir = f"{base_name}{count}"

    bucket.blob(f"{base_dir}/{model_dir}/.keep").upload_from_string("")

    logging.info(f"Created new model directory: {base_dir}/{model_dir}")
    return f"{base_dir}/{model_dir}"


def get_latest_model_from_gcs(base_dir: str = "models"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    base_name = "model"
    count = 0
    model_dir = base_name

    logging.info(
        f"Searching for latest model in bucket: {BUCKET_NAME}, base directory: {base_dir}"
    )

    while True:
        blobs = list(
            bucket.list_blobs(prefix=f"{base_dir}/{model_dir}/", max_results=1)
        )
        if not blobs:
            break
        count += 1
        model_dir = f"{base_name}{count}"

    if count - 1 == 0:
        model_dir = base_name
    else:
        model_dir = f"{base_name}{count-1}"

    logging.info(f"Latest model directory: {base_dir}/{model_dir}")
    print(f"Latest model directory: {base_dir}/{model_dir}")

    model_path = f"{base_dir}/{model_dir}/model.keras"
    labels_path = f"{base_dir}/{model_dir}/labels.txt"

    model_blob = bucket.blob(model_path)
    labels_blob = bucket.blob(labels_path)

    if not model_blob.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not labels_blob.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_model_path = os.path.join(temp_dir, "model.keras")
        model_blob.download_to_filename(temp_model_path)
        model = tf.keras.models.load_model(temp_model_path)
        logging.info(f"Model loaded from {model_path}")

        labels_content = labels_blob.download_as_text()
        labels = {}
        for line in labels_content.split("\n"):
            if line.strip():
                name, index = line.strip().split(": ")
                labels[int(index)] = name
        logging.info(f"Labels loaded from {labels_path}")

    return model, labels


def get_latest_tflite_model_path(base_dir: str = "models"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    base_name = "model"
    count = 0
    model_dir = base_name

    while bucket.blob(f"{base_dir}/{model_dir}/").exists():
        count += 1
        model_dir = f"{base_name}{count}"

    latest_dir = f"{base_name}{count-1}" if count > 1 else base_name
    return f"{base_dir}/{latest_dir}/model.tflite"
