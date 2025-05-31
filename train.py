import os
import torch
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
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-score:  {f1:.4f}")

    return all_preds, all_labels

def visualize_predictions(model, dataloader, device, class_names, num_images=6):
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(15, 8))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//3, 3, images_so_far)
                ax.axis('off')

                img = inputs.cpu().data[j].numpy()
                img = np.transpose(img, (1, 2, 0))
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                ax.imshow(img)
                ax.set_title(f"Pred: {class_names[preds[j]]}\nTrue: {class_names[labels[j]]}")

                if images_so_far == num_images:
                    plt.show()
                    return
    plt.show()

def main():
    base_path = '/Users/proffsenya/Desktop/VKR/pneumonia_classification/PediatricChestX-rayPneumonia'

    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    class_names = train_dataset.classes

    model_names = ['vgg11', 'resnet18', 'densenet69']
    best_acc = 0.0
    best_model_name = None

    for model_name in model_names:
        print(f"\n{'='*30}\nTraining model: {model_name}\n{'='*30}")

        model = get_model(model_name, num_classes=2)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        best_model_path = f'best_{model_name}.pth'

        trained_model = train_model(model, dataloaders, criterion, optimizer,
                                    device, num_epochs=1, save_path=best_model_path)

        print(f"Training finished for {model_name}. Best model saved at {best_model_path}")

        trained_model.load_state_dict(torch.load(best_model_path))

        print(f"Evaluating model {model_name} on test set...")
        preds, labels = test_model(trained_model, test_loader, device)

        correct = sum(p == l for p, l in zip(preds, labels))
        acc = correct / len(labels)
        print(f"Test Accuracy for {model_name}: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name

    print(f"\nBest model overall: {best_model_name} with accuracy {best_acc:.4f}")

    best_model_path = f'best_{best_model_name}.pth'
    best_model = get_model(best_model_name, num_classes=2).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    print(f"\nVisualizing predictions for best model: {best_model_name}")
    visualize_predictions(best_model, test_loader, device, class_names, num_images=6)


if __name__ == '__main__':
    main()
