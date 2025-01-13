import os
import io
from PIL import Image as PilImage
import numpy as np
from google.cloud import translate_v2 as translate
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\\Users\\juanj\\Desktop\\testPrueba\\inspiring-bonus-445203-p0-de2bd3223728.json"

class Modelo:
    def __init__(self, model_path=r"C:\Users\juanj\Desktop\Predicion_Imagenes_Multilabel_enWeb\backend\recursos\optimized_updated_resnet50_model.h5"):
        self.model_path = model_path
        self.model = None
        self.category_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
                               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
                               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
                               'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                               'hair drier', 'toothbrush']

        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise Exception(f"El modelo en {self.model_path} no existe.")
        self.model = load_model(self.model_path)
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy', 'AUC'])

    def predict_image(self, image_file, target_size=(224, 224)):
        # Convertir el archivo cargado a un objeto BytesIO
        image_stream = io.BytesIO(image_file.read())
        image_stream.seek(0)
        image = PilImage.open(image_stream)

        # Asegurarse de que la imagen es RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Redimensionar y convertir la imagen para que sea compatible con el modelo
        image = image.resize(target_size)
        img_array = img_to_array(image)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))

        # Realizar predicción
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Filtrar resultados según el umbral de confianza
        # results = {self.category_names[i]: float(score) for i, score in enumerate(predictions) if score > 0.5}
        predicted_categories = [self.category_names[i] for i, score in enumerate(predictions) if score > 0.5]
        
        return predicted_categories
    
    def translate_text(self, texts, target_language='es'):
        translate_client = translate.Client()
        if isinstance(texts, list):
            translated_texts = []
            for text in texts:
                result = translate_client.translate(text, target_language=target_language)
                translated_texts.append(result['translatedText'])
            return translated_texts
        else:
            result = translate_client.translate(texts, target_language=target_language)
            return result['translatedText']

