import os
import torch
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models import get_model
from utils.train_utils import train_model
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return all_preds, all_labels

def main():
    print("Downloading dataset...")
    base_path = kagglehub.dataset_download("andrewmvd/pediatric-pneumonia-chest-xray")
    if not os.path.isdir(os.path.join(base_path, 'train')):
        for item in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, item, 'train')):
                base_path = os.path.join(base_path, item)
                break

    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')

    # Улучшенные аугментации: RandomResizedCrop и RandomErasing заставляют модель игнорировать фон
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # случайный кроп, чтобы убрать зависимость от краёв
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')  # закрашиваем случайные куски
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Для быстрого обучения используйте subset (например, 1000 изображений)
    # train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    # test_dataset = torch.utils.data.Subset(test_dataset, range(200))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_name = 'vgg11'
    model = get_model(model_name, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_model_path = f'best_{model_name}.pth'
    # Увеличиваем число эпох до 10–20 для лучшего обучения (здесь 1 эпоха для теста, но вы можете изменить)
    train_model(model, {'train': train_loader, 'val': test_loader}, criterion, optimizer, device, num_epochs=1, save_path=best_model_path)

    # Оценка
    model.load_state_dict(torch.load(best_model_path))
    test_model(model, test_loader, device)

if __name__ == '__main__':
    main()