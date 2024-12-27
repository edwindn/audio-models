import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchaudio.models import wav2vec2_base
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


files = [f'data/dataloader_{i}_wav2vec.pth' for i in range(20)] # up to 19
train_loaders = []
for file in files:
    data = torch.load(file)
    train_loaders.append(data.dataset)

train_loader = ConcatDataset(train_loaders)
train_loader = DataLoader(train_loader, shuffle=True, batch_size=16)
test_loader = torch.load('data/dataloader_20_wav2vec.pth')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 8, 4, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 8, 4, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        ).to(device)
    
    def forward(self, x):
        x = x.to(device)
        x = x.squeeze().unsqueeze(1)
        return self.main(x)
    
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

def train():
    for epoch in range(20):
        torch.cuda.empty_cache()

        print(f'Epoch {epoch}')
        epoch_loss = 0
        net.train()
        for batch in tqdm(train_loader):
            audio, label = batch
            audio = audio.to(device)
            label = label.to(device)
            preds = net(audio)
            loss = loss_fn(preds, label)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Loss: {epoch_loss / len(train_loader)}')
        net.main.eval()
        correct = 0
        total = 0
        for batch in test_loader:
            audio, label = batch
            audio = audio.to(device)
            label = label.to(device)
            preds = net(audio)
            preds = torch.argmax(preds, dim=1)
            correct += (preds == label).sum().item()
            total += len(label)
        print(f'Accuracy: {correct/total*100}%')

train()
