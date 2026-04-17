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

# тепловые карты
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def main():
    print("Проверяем датасет...")
    base_path = kagglehub.dataset_download("andrewmvd/pediatric-pneumonia-chest-xray")
    
    if not os.path.isdir(os.path.join(base_path, 'train')):
        for item in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, item, 'train')):
                base_path = os.path.join(base_path, item)
                break
                
    test_dir = os.path.join(base_path, 'test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")


    model_name = 'resnet18' 
    model_path = f'best_{model_name}.pth'
    
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл {model_path} не найден!")
        print("Сначала дождитесь окончания работы файла train.py, чтобы нейросеть обучилась и сохранила свои 'мозги'.")
        return

    print("Загружаем обученную нейросеть...")
    model = get_model(model_name, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if model_name == 'resnet18':
        target_layers = [model.layer4[-1]]
    elif model_name == 'vgg11':
        target_layers = [model.features[-1]]
    else:
        target_layers = [model.features[-1]] # для densenet

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
    class_names = test_dataset.classes

    print("Выбираем случайные снимки и строим тепловые карты...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 6 случайных снимков (как больных, так и здоровых)
    pneumonia_idx = test_dataset.class_to_idx['PNEUMONIA']
    all_samples = list(range(len(test_dataset)))
    selected_indices = np.random.choice(all_samples, 6, replace=False)

    for i, idx in enumerate(selected_indices):
        img_tensor, label = test_dataset[idx]
        img_path = test_dataset.samples[idx][0]
        
        input_tensor = img_tensor.unsqueeze(0).to(device)
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds[0].item()]
        true_class = class_names[label]

        # тепловая карта (всегда ищем признаки пневмонии чтобы понять логику сети)
        targets = [ClassifierOutputTarget(pneumonia_idx)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = np.where(grayscale_cam < 0.2, 0, grayscale_cam)

        # отрисовка тепловой карты на оригинальном изображении
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, (224, 224))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = np.float32(original_img) / 255.0

        visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)

        axes[i].imshow(visualization)
        
        title_text = f"Прогноз: {pred_class}\nНа самом деле: {true_class}"
        title_obj = axes[i].set_title(title_text, fontsize=14)
        
        if pred_class == true_class:
            title_obj.set_color('green')
        else:
            title_obj.set_color('red')
            
        axes[i].axis('off')

    plt.suptitle("Зоны внимания нейросети", fontsize=18)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()