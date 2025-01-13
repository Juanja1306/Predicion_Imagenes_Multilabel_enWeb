from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
#import matplotlib.pyplot as plt
from PIL import Image, ExifTags

class MLC(nn.Module):
    def __init__(self, num_classes):
        super(MLC, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet50(weights=None).to(self.device)
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_classes).to(self.device)
        self.load_model_weights(r"C:\Users\juanj\Desktop\Predicion_Imagenes_Multilabel_enWeb\backend\recursos\modelo_pesos_todas_imagenes.pth")

    def forward(self, x):
        x = self.resnet(x)
        return x

    def load_model_weights(self, weight_path):
        state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
        new_state_dict = {k.replace('resnet.', ''): v for k, v in state_dict.items()}
        self.resnet.load_state_dict(new_state_dict, strict=False)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def adjust_image_orientation(image):
    orientation_tag = next((tag for tag, value in ExifTags.TAGS.items() if value == 'Orientation'), None)
    if orientation_tag and hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif and orientation_tag in exif:
            orientation = exif[orientation_tag]
            rotations = {3: 180, 6: 270, 8: 90}
            image = image.rotate(rotations.get(orientation, 0), expand=True)
    return image


def predict_and_show(image_path, model):
    image = Image.open(image_path)
    image = adjust_image_orientation(image)
    image = image.convert('RGB')
    image_transformed = transform(image).unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model(image_transformed)
    probs = torch.sigmoid(outputs).cpu().numpy()
    predicted_ids = [index_to_id[i] for i, p in enumerate(probs[0]) if p > 0.5]
    labels = [category_names[cat_id] for cat_id in predicted_ids]
   
    return labels

# Configuraciones adicionales
category_names = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 
                   21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 
                   34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 
                   42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
                   52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 
                   74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 
                   84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}
index_to_id =  {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 
                11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 
                21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 
                31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 
                41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 
                51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 
                61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 
                71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
                }


# # Uso del modelo
# model = MLC(num_classes=78)
# image_path = "C:/Users/juanj/Pictures/fotos/20221130_132637.jpg"
# predicted_labels = predict_and_show(image_path, model)
# print("Etiquetas predichas:")
# print(predicted_labels)
