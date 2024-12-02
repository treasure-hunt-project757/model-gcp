# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def prepare_data(directory):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = train_datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_data = train_datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    return train_data, val_data
