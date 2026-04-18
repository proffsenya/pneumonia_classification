import base64
import io
import os
import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image
import cv2
import requests
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from models import get_model

app = Flask(__name__)

MODEL_NAME = 'vgg11'
MODEL_PATH = 'best_vgg11.pth'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}. Сначала запустите train.py.")

model = get_model(MODEL_NAME, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

cam = GradCAMPlusPlus(model=model, target_layers=[model.features[-1]])

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def np_image_to_base64(image_np: np.ndarray) -> str:
    image_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_uint8)
    return image_to_base64(image)


def is_ollama_available() -> bool:
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False


# ---------- МАСКА ЛЁГКИХ (анатомически обоснованная) ----------
def create_lung_mask(shape: tuple) -> np.ndarray:
    """Маска двух лёгких + центральная область (трахея/средостение)."""
    mask = np.zeros(shape, dtype=np.float32)
    h, w = shape
    center_h, center_w = h // 2, w // 2
    y, x = np.ogrid[:h, :w]

    # Левое лёгкое
    left = ((x - w * 0.35) ** 2 / (w * 0.2) ** 2 + (y - center_h) ** 2 / (h * 0.35) ** 2) <= 1
    # Правое лёгкое
    right = ((x - w * 0.65) ** 2 / (w * 0.2) ** 2 + (y - center_h) ** 2 / (h * 0.35) ** 2) <= 1
    # Центральная область (соединяет лёгкие, но не фон)
    center = (x >= w * 0.45) & (x <= w * 0.55) & (y >= h * 0.4) & (y <= h * 0.6)

    mask[left | right | center] = 1.0

    # Морфологическое закрытие (убираем дыры)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def filter_heatmap_by_lungs(grayscale_cam: np.ndarray) -> np.ndarray:
    """Применяет маску, порог и нормализует тепловую карту в области лёгких."""
    mask = create_lung_mask(grayscale_cam.shape)
    masked = grayscale_cam * mask

    # Убираем шум (значения < 0.15)
    threshold = 0.15
    masked = np.where(masked < threshold, 0.0, masked)

    # Нормализуем в [0,1] по оставшимся активным пикселям
    if masked.max() > 0:
        masked = (masked - masked.min()) / (masked.max() - masked.min() + 1e-8)
    return masked

# ---------- ЛОКАЛЬНЫЙ ДИАГНОЗ (без Ollama) ----------
def interpret_heatmap(pred_class: str, true_class: str, grayscale_cam: np.ndarray) -> str:
    cam_filtered = filter_heatmap_by_lungs(grayscale_cam)
    max_score = float(np.max(cam_filtered))
    mean_score = float(np.mean(cam_filtered[cam_filtered > 0])) if np.any(cam_filtered > 0) else 0.0
    hot_ratio = float(np.mean(cam_filtered > 0.5)) if cam_filtered.size > 0 else 0.0

    if pred_class == 'PNEUMONIA':
        if max_score > 0.7 or hot_ratio > 0.12:
            confidence = 'высокой'
        elif max_score > 0.4 or hot_ratio > 0.06:
            confidence = 'средней'
        else:
            confidence = 'низкой'
        diagnosis = (f"Предварительный диагноз: пневмония с {confidence} уверенностью. "
                     f"Области внимания (только лёгкие): max={max_score:.2f}, avg={mean_score:.2f}.")
    else:
        diagnosis = (f"Предварительный диагноз: норма. Явных признаков пневмонии не обнаружено. "
                     f"Интенсивность в лёгких: max={max_score:.2f}, avg={mean_score:.2f}.")

    if true_class != 'UNKNOWN':
        diagnosis += f" Истинный класс: {true_class}."
    return diagnosis

# ---------- OLLAMA (опционально) ----------
def build_ollama_prompt(pred_class: str, grayscale_cam: np.ndarray, true_class: str = 'UNKNOWN') -> str:
    cam_filtered = filter_heatmap_by_lungs(grayscale_cam)
    max_score = float(np.max(cam_filtered))
    mean_score = float(np.mean(cam_filtered[cam_filtered > 0])) if np.any(cam_filtered > 0) else 0.0
    hot_ratio = float(np.mean(cam_filtered > 0.5))
    return (f"Ты медицинский ассистент. Всегда отвечай только на русском языке. Не используй английский. "
            f"На основе анализа рентгеновского снимка и тепловой карты: предсказан класс {pred_class}, истинный класс {true_class}. "
            f"Параметры тепловой карты (только в области лёгких): максимум {max_score:.2f}, среднее {mean_score:.2f}, доля горячих пикселей {hot_ratio:.2f}. "
            "Дай предварительный диагноз. Начни с 'ПЕРВИЧНОЕ ЗАКЛЮЧЕНИЕ:'. Опиши тип пневмонии (если есть), локализацию (левое/правое лёгкое, доли), стадию (острая/хроническая), степень тяжести. Будь кратким, но информативным. Это не замена врачу.")

def generate_ollama_report(pred_class: str, true_class: str, grayscale_cam: np.ndarray) -> str:
    try:
        response = requests.post('http://localhost:11434/api/generate',
                                 json={'model': OLLAMA_MODEL, 'prompt': build_ollama_prompt(pred_class, grayscale_cam, true_class), 'stream': False},
                                 timeout=60)
        if response.status_code == 200:
            output = response.json().get('response', '').strip()
            if output:
                return output
    except Exception:
        pass
    return interpret_heatmap(pred_class, true_class, grayscale_cam)  # fallback

# ---------- ВИЗУАЛИЗАЦИЯ ТЕПЛОВОЙ КАРТЫ ----------
def make_heatmap_image(resized: Image.Image, grayscale_cam: np.ndarray, overlay: bool = True) -> np.ndarray:
    # Применяем маску лёгких к тепловой карте
    grayscale_cam_masked = filter_heatmap_by_lungs(grayscale_cam)

    # Цветовая карта Jet
    heatmap = np.uint8(255 * grayscale_cam_masked)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if overlay:
        image_np = np.array(resized).astype(np.float32) / 255.0
        gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

        # Альфа-канал на основе тепловой карты (сильнее там, где выше активация)
        alpha = np.expand_dims(grayscale_cam_masked * 0.8 + 0.2, axis=2)
        heatmap_norm = heatmap.astype(np.float32) / 255.0
        overlay_img = gray_rgb * (1.0 - alpha) + heatmap_norm * alpha
        overlay_img = np.clip(overlay_img, 0, 1)
        return overlay_img
    return heatmap.astype(np.float32) / 255.0

# ---------- FLASK ----------
def prepare_input(image_file):
    original = Image.open(image_file).convert('RGB')
    resized = original.resize((224, 224))
    tensor = TRANSFORM(resized).unsqueeze(0).to(DEVICE)
    return original, resized, tensor

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    original_image = None
    heatmap_image = None
    diagnosis = None
    ollama_status = is_ollama_available()

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            original, resized, input_tensor = prepare_input(file)
            original_image = image_to_base64(original)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                pred_class = CLASS_NAMES[predicted.item()]

            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[ClassifierOutputTarget(CLASS_NAMES.index('PNEUMONIA'))],
                                aug_smooth=True, eigen_smooth=True)
            grayscale_cam = grayscale_cam[0, :]

            heatmap_rgb = make_heatmap_image(resized, grayscale_cam, overlay=True)
            heatmap_image = np_image_to_base64(heatmap_rgb)

            diagnosis = generate_ollama_report(pred_class, 'UNKNOWN', grayscale_cam)
            result = {'prediction': pred_class, 'model': MODEL_NAME}
        else:
            result = {'error': 'Файл не загружен'}

    return render_template('index.html', result=result, original_image=original_image,
                           heatmap_image=heatmap_image, diagnosis=diagnosis,
                           ollama_status=ollama_status, model_name=MODEL_NAME, ollama_model=OLLAMA_MODEL)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)