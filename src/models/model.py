import time
from fastapi import HTTPException
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np
from src.utils.data_utils import num_of_classes, remove_ds_store
from src.utils.training_utils import prepare_data
from src.utils.api_utils import fetch_objects_from_server
from src.utils.gcs_utils import (
    create_model_directory,
    download_images,
    get_latest_model_from_gcs,
    get_latest_tflite_model_path,
    save_model_to_gcs,
)
import shutil
from functools import lru_cache

THRESHOLD = 0.4
EPOCHS = 10


class ModelHandler:
    def __init__(self):
        self.dataset_dir = None
        self.models_dir = "models"
        self.model = None
        self.last_update_time = 0
        self.cache_duration = 30 * 60

    @lru_cache(maxsize=1)
    def get_cached_model(self):
        current_time = time.time()
        if current_time - self.last_update_time > self.cache_duration:
            self.model, self.labels = get_latest_model_from_gcs(self.models_dir)
        self.last_update_time = current_time
        return self.model, self.labels

    # @lru_cache(maxsize=1)
    # def get_cached_model(self):
    #     current_time = time.time()
    #     if current_time - self.last_update_time > self.cache_duration:
    #         self.model, self.labels = get_latest_model_from_gcs(self.models_dir)
    #         self.last_update_time = current_time
    #     return self.model, self.labels

    def build_model(self, num_classes):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
        base_model.trainable = False
        global_average_layer = layers.GlobalAveragePooling2D()
        prediction_layer = layers.Dense(num_classes, activation="softmax")

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = global_average_layer(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def retrain(self):
        objects = fetch_objects_from_server()
        if not objects:
            raise HTTPException(
                status_code=500, detail="Failed to fetch objects from server"
            )
        self.dataset_dir = download_images(objects)
        remove_ds_store(self.dataset_dir)
        num_classes = num_of_classes(self.dataset_dir)
        self.model = self.build_model(num_classes)
        train_data, val_data = prepare_data(self.dataset_dir)
        self.model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
        print("class indices are    ", train_data.class_indices)
        self.save_model(train_data.class_indices)
        shutil.rmtree(self.dataset_dir)

    def save_model(self, class_indices):
        model_dir = create_model_directory(self.models_dir)
        gcs_path = save_model_to_gcs(self.model, class_indices, model_dir)
        # self.model.summary()
        print(f"Model saved to: {gcs_path}")

    def predict(self, image_raw):
        input_arr = np.expand_dims(image_raw, axis=0).astype(np.float32)
        # model, labels = get_latest_model_from_gcs(self.models_dir)
        model, labels = self.get_cached_model()
        print("labels are ", labels)

        threshold = THRESHOLD
        if len(labels) < 7:
            threshold = 0.7
        # keras prediction

        keras_prediction = model.predict(input_arr)
        print("Keras prediction: ", keras_prediction)
        keras_max_prob = tf.reduce_max(keras_prediction).numpy()
        print("Keras max probability: ", keras_max_prob)
        keras_class_idx = tf.argmax(keras_prediction, axis=1).numpy()[0]
        keras_result = (
            labels[keras_class_idx] if keras_max_prob >= threshold else "Unknown"
        )

        return keras_result, float(keras_max_prob)

    def get_latest_tflite_model_dir(self):
        return get_latest_tflite_model_path(self.models_dir)

    # def predict(self, image_raw):
    #     input_arr = np.expand_dims(image_raw, axis=0).astype(np.float32)
    #     model, labels = self.get_latest_model(self.models_dir)
    #     print("labels are ", labels)

    #     # keras prediction
    #     keras_prediction = model.predict(input_arr)
    #     print("Keras prediction: ", keras_prediction)
    #     keras_max_prob = tf.reduce_max(keras_prediction).numpy()
    #     print("Keras max probability: ", keras_max_prob)
    #     keras_class_idx = tf.argmax(keras_prediction, axis=1).numpy()[0]
    #     keras_result = (
    #         labels[keras_class_idx] if keras_max_prob >= THRESHOLD else "Unknown"
    #     )

    #     return keras_result, float(keras_max_prob)

    # def get_latest_tflite_model_dir(self):
    #     base_name = "model"
    #     count = 0
    #     model_dir = base_name

    #     while os.path.exists(os.path.join(self.models_dir, model_dir)):
    #         count += 1
    #         model_dir = f"{base_name}{count}"

    #     latest_dir = f"{base_name}{count-1}"
    #     return os.path.join(self.models_dir, latest_dir)

    ######################################
    # def create_model_directory(self, directory):
    #     base_name = "model"
    #     count = 0
    #     model_dir = base_name

    #     while os.path.exists(os.path.join(directory, model_dir)):
    #         count += 1
    #         model_dir = f"{base_name}{count}"

    #     os.makedirs(os.path.join(directory, model_dir))
    #     return model_dir

    # def get_latest_model(self, directory):
    #     base_name = "model"
    #     count = 0
    #     model_dir = base_name
    #     print("in gettin model ")
    #     while os.path.exists(os.path.join(directory, model_dir)):
    #         print("model dir is ", model_dir)
    #         count += 1
    #         model_dir = f"{base_name}{count}"

    #     if count - 1 == 0:
    #         print("in if ", count)
    #         model_dir = base_name
    #     else:
    #         print("in else ", count)
    #         model_dir = f"{base_name}{(count-1)}"

    #     print("finished with count ", (count - 1))
    #     print("finished - model dir is  ", model_dir)

    #     model_path = os.path.join(directory, model_dir, "model.keras")
    #     labels_path = os.path.join(directory, model_dir, "labels.txt")

    #     if os.path.exists(model_path):
    #         print("model exists")
    #         latest_model = tf.keras.models.load_model(model_path)
    #     else:
    #         print("model does not exist")
    #         raise FileNotFoundError(f"Model file not found in directory: {model_path}")

    #     labels = {}
    #     if os.path.exists(labels_path):
    #         with open(labels_path, "r") as file:
    #             for line in file:
    #                 name, index = line.strip().split(": ")
    #                 labels[int(index)] = name
    #     else:
    #         raise FileNotFoundError(
    #             f"Labels file not found in directory: {labels_path}"
    #         )

    #     return latest_model, labels
