import base64
import io
import os
import json
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, Response
from PIL import Image
import cv2
import requests
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms, models

app = Flask(__name__)

# ---------- Конфигурация ----------
MODEL_PATH = 'best_vgg11.pth'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OLLAMA_MODEL_1 = 'mistral'
OLLAMA_MODEL_2 = 'llama2'

# ---------- Создание модели VGG11 (без внешнего models.py) ----------
def get_model(num_classes=2):
    model = models.vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

# Загрузка весов
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл весов {MODEL_PATH} не найден. Сначала обучите модель.")
model = get_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# GradCAM на последнем свёрточном слое
target_layers = [model.features[-1]]
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

# Трансформации
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- Вспомогательные функции ----------
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
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        return response.status_code == 200
    except:
        return False

def query_ollama_model(model_name: str, prompt: str, timeout: int = None) -> str:
    if timeout is None:
        timeout = 180 if model_name == 'llama2' else 120
    try:
        response = requests.post('http://localhost:11434/api/generate',
                                 json={'model': model_name, 'prompt': prompt, 'stream': False},
                                 timeout=timeout)
        if response.status_code == 200:
            output = response.json().get('response', '').strip()
            return output if output else f"Ошибка: {model_name} вернула пустой ответ"
    except requests.exceptions.Timeout:
        return f"Ошибка: {model_name} - timeout"
    except Exception as e:
        return f"Ошибка: {str(e)}"
    return f"Ошибка: не удалось подключиться к {model_name}"

# ---------- Анатомическая маска лёгких ----------
def create_lung_mask(shape: tuple) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    h, w = shape
    center_h, center_w = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    left = ((x - w * 0.35) ** 2 / (w * 0.2) ** 2 + (y - center_h) ** 2 / (h * 0.35) ** 2) <= 1
    right = ((x - w * 0.65) ** 2 / (w * 0.2) ** 2 + (y - center_h) ** 2 / (h * 0.35) ** 2) <= 1
    center = (x >= w * 0.45) & (x <= w * 0.55) & (y >= h * 0.4) & (y <= h * 0.6)
    mask[left | right | center] = 1.0
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def filter_heatmap_by_lungs(grayscale_cam: np.ndarray) -> np.ndarray:
    mask = create_lung_mask(grayscale_cam.shape)
    masked = grayscale_cam * mask
    threshold = 0.15
    masked = np.where(masked < threshold, 0.0, masked)
    if masked.max() > 0:
        masked = (masked - masked.min()) / (masked.max() - masked.min() + 1e-8)
    return masked

# ---------- Построение тепловой карты ----------
def make_heatmap_image(resized: Image.Image, grayscale_cam: np.ndarray, overlay: bool = True) -> np.ndarray:
    grayscale_cam_masked = filter_heatmap_by_lungs(grayscale_cam)
    heatmap = np.uint8(255 * grayscale_cam_masked)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if overlay:
        image_np = np.array(resized).astype(np.float32) / 255.0
        gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        alpha = np.expand_dims(grayscale_cam_masked * 0.8 + 0.2, axis=2)
        heatmap_norm = heatmap.astype(np.float32) / 255.0
        overlay_img = gray_rgb * (1.0 - alpha) + heatmap_norm * alpha
        overlay_img = np.clip(overlay_img, 0, 1)
        return overlay_img
    return heatmap.astype(np.float32) / 255.0

# ---------- Построение промпта для LLM ----------
def build_neural_prompt(pred_class: str, grayscale_cam: np.ndarray) -> str:
    mask = create_lung_mask(grayscale_cam.shape)
    masked = grayscale_cam * mask
    active_pixels = masked[masked > 0]
    if len(active_pixels) > 0:
        max_score = float(np.max(masked))
        mean_score = float(np.mean(active_pixels))
        threshold_high = 0.5 * max_score
        hot_ratio = float(np.sum(masked > threshold_high) / np.sum(mask > 0))
    else:
        max_score = mean_score = hot_ratio = 0.0

    if max_score > 0.7:
        intensity_desc = "ВЫСОКАЯ активация - явные признаки поражения"
    elif max_score > 0.4:
        intensity_desc = "СРЕДНЯЯ активация - вероятные признаки"
    else:
        intensity_desc = "НИЗКАЯ активация - норма или минимальные изменения"

    if hot_ratio > 0.15:
        area_desc = "большой области лёгких поражена"
    elif hot_ratio > 0.05:
        area_desc = "локальные очаги поражения"
    else:
        area_desc = "минимальные признаки или норма"

    return (f"АНАЛИЗ РЕНТГЕН-СНИМКА:\n"
            f"Предсказание нейросети: {pred_class}\n\n"
            f"ПАРАМЕТРЫ ТЕПЛОВОЙ КАРТЫ (область лёгких):\n"
            f"- Максимум активации: {max_score:.3f} ({intensity_desc})\n"
            f"- Средняя интенсивность: {mean_score:.3f}\n"
            f"- Процент горячих пикселей: {hot_ratio*100:.1f}% ({area_desc})\n\n"
            f"ИНТЕРПРЕТАЦИЯ:\n"
            f"Если max > 0.7 = сильное поражение\n"
            f"Если max 0.4-0.7 = среднее поражение\n"
            f"Если max < 0.4 = норма или ранние признаки")

# ---------- Flask routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_stream', methods=['POST'])
def analyze_stream():
    file = request.files.get('file')
    if not file or not file.filename:
        return Response("data: " + json.dumps({'type': 'error', 'message': 'Файл не загружен'}) + "\n\n",
                        mimetype='text/event-stream')

    file_content = file.read()
    file_name = file.filename

    def generate():
        try:
            file_obj = io.BytesIO(file_content)
            file_obj.name = file_name

            original, resized, input_tensor = prepare_input(file_obj)
            original_image = image_to_base64(original)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                pred_class = CLASS_NAMES[predicted.item()]

            yield f"data: {json.dumps({'type': 'prediction', 'prediction': pred_class})}\n\n"

            grayscale_cam_raw = cam(input_tensor=input_tensor,
                                    targets=[ClassifierOutputTarget(CLASS_NAMES.index(pred_class))],
                                    aug_smooth=True, eigen_smooth=True)[0, :]

            heatmap_rgb = make_heatmap_image(resized, grayscale_cam_raw, overlay=True)
            heatmap_image = np_image_to_base64(heatmap_rgb)

            yield f"data: {json.dumps({'type': 'heatmap', 'original': original_image, 'heatmap': heatmap_image})}\n\n"

            if not is_ollama_available():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Ollama недоступна'})}\n\n"
                return

            context = build_neural_prompt(pred_class, grayscale_cam_raw)

            # Mistral
            prompt1 = (f"Ты опытный врач-рентгенолог. Анализируешь рентгеновский снимок грудной клетки.\n{context}\n\n"
                       f"Дай СВОЁ МЕДИЦИНСКОЕ МНЕНИЕ (2-3 предложения) на русском языке об этом снимке.\n"
                       f"Основывайся на показателях активации (если max высокий = проблема, если низкий = норма).\n"
                       f"Чётко скажи: есть ли признаки пневмонии или нет.")
            response1 = query_ollama_model(OLLAMA_MODEL_1, prompt1)
            yield f"data: {json.dumps({'type': 'model1', 'response': response1})}\n\n"

            # Llama2
            prompt2 = (f"ВНИМАНИЕ! Ты ОБЯЗАТЕЛЬНО должен отвечать ТОЛЬКО НА РУССКОМ ЯЗЫКЕ! Не используй английский!\n"
                       f"Ты опытный врач-рентгенолог. Анализируешь тот же рентгеновский снимок грудной клетки.\n{context}\n\n"
                       f"Твой коллега сказал: \"{response1}\"\n\n"
                       f"СОГЛАСЕН ЛИ ТЫ с этим диагнозом? Дай СВОЁ МНЕНИЕ (2-3 предложения) НА РУССКОМ ЯЗЫКЕ.\n"
                       f"Если согласен - подтверди. Если не согласен - объясни почему и дай свой диагноз.\n"
                       f"Обсуждайте на основе параметров (max активация, процент горячих пикселей).\n"
                       f"ТОЛЬКО РУССКИЙ ЯЗЫК! БЕЗ АНГЛИЙСКОГО!")
            response2 = query_ollama_model(OLLAMA_MODEL_2, prompt2)
            yield f"data: {json.dumps({'type': 'model2', 'response': response2})}\n\n"

            # Финальный диагноз
            prompt3 = (f"Ты главный врач-консультант. Двое твоих коллег обсудили рентгеновский снимок.\n{context}\n\n"
                       f"Коллега 1 ({OLLAMA_MODEL_1}): \"{response1}\"\n\n"
                       f"Коллега 2 ({OLLAMA_MODEL_2}): \"{response2}\"\n\n"
                       f"На основе их мнений и параметров активации, дай ФИНАЛЬНЫЙ ДИАГНОЗ (одной строкой).\n"
                       f"Формат: 'ДИАГНОЗ: [результат]' или 'НОРМА' или 'ПНЕВМОНИЯ: [описание]'")
            final_diagnosis = query_ollama_model(OLLAMA_MODEL_1, prompt3)
            yield f"data: {json.dumps({'type': 'final', 'response': final_diagnosis})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

def prepare_input(image_file):
    original = Image.open(image_file).convert('RGB')
    resized = original.resize((224, 224))
    tensor = TRANSFORM(resized).unsqueeze(0).to(DEVICE)
    return original, resized, tensor

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)