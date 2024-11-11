# tune.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm

DATA_FINE_TUNING = 'data_fine_tuning'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'fine_tuned_model.pth'

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

class FineTuner:
    def __init__(self, data_dir, batch_size, epochs, lr, device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.dataset = FaceDataset(data_dir, transform=self.transform)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.model = InceptionResnetV1(pretrained='vggface2').to(self.device)
        self.model.train()
        self.num_classes = len(self.dataset.dataset.classes)
        self.classifier = nn.Linear(self.model.last_linear.in_features, self.num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.classifier.parameters()), lr=lr)

    def train(self, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for imgs, labels in tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                embeddings = self.model(imgs)
                outputs = self.classifier(embeddings)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            avg_loss = running_loss / len(self.loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'classes': self.dataset.dataset.classes
        }, MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fine_tuner = FineTuner(
        data_dir=DATA_FINE_TUNING,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device
    )
    fine_tuner.train(EPOCHS)

if __name__ == '__main__':
    main()
