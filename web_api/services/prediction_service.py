import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (300, 300)
MODEL_PATH = '../best_skin_model_v2.h5'
CLASS_NAMES_PATH = '../class_names.txt'


class PredictionService:
    def __init__(self):
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.num_classes = len(self.class_names)
        self.model = self._build_model()
        self.model.load_weights(MODEL_PATH)

    def _build_model(self):
        base_model = tf.keras.applications.EfficientNetB3(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights=None
        )
        inputs = layers.Input(shape=IMG_SIZE + (3,))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return models.Model(inputs, outputs)

    def _preprocess_image(self, image_path):
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array

    def predict(self, file_path):
        img_tensor = self._preprocess_image(file_path)
        predictions = self.model.predict(img_tensor)

        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_disease = self.class_names[predicted_class_idx]

        return {
            "disease": predicted_disease,
            "confidence": round(confidence * 100, 2)
        }


prediction_service = PredictionService()