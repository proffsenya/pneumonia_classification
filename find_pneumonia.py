import os
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from torchvision import transforms, datasets
from models import get_model
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------- МАСКА ЛЁГКИХ (та же, что в app.py) ----------
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

def interpret_heatmap(pred_class, true_class, grayscale_cam):
    cam_filtered = filter_heatmap_by_lungs(grayscale_cam)
    max_score = float(np.max(cam_filtered))
    mean_score = float(np.mean(cam_filtered[cam_filtered > 0])) if np.any(cam_filtered > 0) else 0.0
    hot_ratio = float(np.mean(cam_filtered > 0.5))

    if pred_class == 'PNEUMONIA':
        if max_score > 0.7 or hot_ratio > 0.12:
            confidence = 'высокой'
        elif max_score > 0.4 or hot_ratio > 0.06:
            confidence = 'средней'
        else:
            confidence = 'низкой'
        diagnosis = (f"Предварительный диагноз: пневмония с {confidence} уверенностью. "
                     f"Активные области в лёгких (max={max_score:.2f}, avg={mean_score:.2f}).")
    else:
        diagnosis = (f"Предварительный диагноз: норма. Явных признаков пневмонии нет. "
                     f"Активность в лёгких: max={max_score:.2f}, avg={mean_score:.2f}.")
    if pred_class != true_class:
        diagnosis += f" Истинный класс: {true_class}."
    return diagnosis

def generate_ollama_report(pred_class, true_class, grayscale_cam):
    # Упрощённая версия – просто возвращаем локальный диагноз (чтобы не зависеть от Ollama)
    return interpret_heatmap(pred_class, true_class, grayscale_cam)

def main():
    print("Загрузка датасета...")
    base_path = kagglehub.dataset_download("andrewmvd/pediatric-pneumonia-chest-xray")
    if not os.path.isdir(os.path.join(base_path, 'train')):
        for item in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, item, 'train')):
                base_path = os.path.join(base_path, item)
                break
    test_dir = os.path.join(base_path, 'test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")

    model_name = 'vgg11'
    model_path = f'best_{model_name}.pth'
    if not os.path.exists(model_path):
        print(f"ОШИБКА: {model_path} не найден. Запустите train.py.")
        return

    model = get_model(model_name, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cam = GradCAMPlusPlus(model=model, target_layers=[model.features[-1]])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    class_names = test_dataset.classes

    # Выбираем 6 случайных снимков
    indices = np.random.choice(len(test_dataset), 6, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img_tensor, label = test_dataset[idx]
        img_path = test_dataset.samples[idx][0]

        input_tensor = img_tensor.unsqueeze(0).to(device)
        outputs = model(input_tensor)
        pred = class_names[torch.argmax(outputs, 1).item()]
        true = class_names[label]

        # Тепловая карта
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(class_names.index('PNEUMONIA'))],
                            aug_smooth=True, eigen_smooth=True)[0, :]

        # Фильтрация по лёгким
        grayscale_cam = filter_heatmap_by_lungs(grayscale_cam)

        # Визуализация
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (224, 224))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) / 255.0

        # Показываем тепловую карту поверх снимка
        heatmap = np.uint8(255 * grayscale_cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        alpha = np.expand_dims(grayscale_cam * 0.7 + 0.3, axis=2)
        overlay = original_img * (1 - alpha) + heatmap * alpha
        overlay = np.clip(overlay, 0, 1)

        diagnosis = interpret_heatmap(pred, true, grayscale_cam)
        axes[i].imshow(overlay)
        axes[i].set_title(diagnosis, fontsize=9)
        axes[i].axis('off')
        print(f"{i+1}. {diagnosis}")

    plt.suptitle("Тепловые карты (только лёгкие) + диагноз", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()