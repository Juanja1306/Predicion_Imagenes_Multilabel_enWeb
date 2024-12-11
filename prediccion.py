from torchvision import datasets, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

class MLC(nn.Module):
    def __init__(self, num_classes):
        super(MLC, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

#Cargamos modelo
model = models.resnet18(weights=None)

# Definir los dispositivos (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = MLC(num_classes=78)
model.resnet
model.to(device)

# Cargar el modelo en CPU
state_dict = torch.load(r"./backend/recursos/modelo_pesos1.pth", map_location=torch.device(device), weights_only=True)
new_state_dict = {k.replace('resnet.', ''): v for k, v in state_dict.items()}
model.resnet.load_state_dict(new_state_dict, strict=False)
model.eval()  # Poner el modelo en modo de evaluación

# Crear un mapeo de índice a ID de categoría
id_to_index = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 81: 70, 82: 71, 84: 72, 85: 73,
    86: 74, 87: 75, 88: 76, 90: 77
}

index_to_id =  {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19,
    18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28,
    26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38,
    34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47,
    42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63,
    58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75,
    66: 76, 67: 77, 68: 78, 69: 79, 70: 81, 71: 82, 72: 84, 73: 85,
    74: 86, 75: 87, 76: 88, 77: 90
}

category_names = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
    33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 90: 'toothbrush'
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def adjust_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # casos en que no hay etiquetas EXIF
        pass
    return image

def predict_and_show(image_path, model):
    model.eval()  # Poner el modelo en modo evaluación
    image = Image.open(image_path)  # Cargar la imagen con PIL
    image = adjust_image_orientation(image)  # Ajustar la orientación de la imagen
    image_transformed = transform(image).unsqueeze(0).to(device)  # Aplicar transformaciones
    
    with torch.no_grad():
        outputs = model(image_transformed)  # Inferencia
        
    probs = torch.sigmoid(outputs).cpu().numpy()  # Aplicar sigmoid para obtener probabilidades
    predicted_indices = [i for i, p in enumerate(probs[0]) if p > 0.5]  # Obtener los índices predichos
    predicted_ids = [index_to_id[i] for i in predicted_indices]  # Convertir índices a IDs de categoría
    labels = [category_names[cat_id] for cat_id in predicted_ids]  # Obtener los nombres de las categorías
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)  # Mostrar la imagen ajustada
    plt.axis('off')
    plt.title(f"Predicciones: {', '.join(labels)}", fontsize=12)
    plt.show()
    
    return labels

image_path = r"C:\Users\juanj\Pictures\fotos\20221130_132637.jpg"
predicted_labels = predict_and_show(image_path, model)
print("Etiquetas predichas:")
print(predicted_labels)

