import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader
from torchaudio.models import wav2vec2_base
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm

def load_wav_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

yes_data = []
no_data = []
sr = 44100

lens = []
for i in range(370):
    no, _ = load_wav_file(f'yesno_voice_recognition/train/no{i}.wav')
    no = librosa.resample(no, orig_sr=44100, target_sr=16000) # optional for direct model
    lens.append(no.shape[0])
    no_data.append((no, 0))
    yes, _ = load_wav_file(f'yesno_voice_recognition/train/yes{i}.wav')
    yes = librosa.resample(yes, orig_sr=44100, target_sr=16000) # optional for direct model
    lens.append(yes.shape[0])
    yes_data.append((yes, 1))

test_yes = []
test_no = []

for i in range(370, 400):
    no, _ = load_wav_file(f'yesno_voice_recognition/train/no{i}.wav')
    no = librosa.resample(no, orig_sr=44100, target_sr=16000) # optional for direct model
    test_no.append((no, 0))
    yes, _ = load_wav_file(f'yesno_voice_recognition/train/yes{i}.wav')
    yes = librosa.resample(yes, orig_sr=44100, target_sr=16000) # optional for direct model
    test_yes.append((yes, 1))

data = yes_data + no_data

fixed_len  = 60000
for i, (d, l) in enumerate(data):
    if d.shape[0] < fixed_len:
        d = np.pad(d, (0, fixed_len - d.shape[0]), 'constant')
    else:
        d = d[:fixed_len]
    data[i] = (d, l)

test_data = test_yes + test_no
fixed_len  = 60000
for i, (d, l) in enumerate(test_data):
    if d.shape[0] < fixed_len:
        d = np.pad(d, (0, fixed_len - d.shape[0]), 'constant')
    else:
        d = d[:fixed_len]
    test_data[i] = (d, l)

data = [(torch.tensor(d, dtype=torch.float32).unsqueeze(0), torch.tensor(l)) for d, l in data]
dataloader = DataLoader(data, shuffle=True, batch_size=4)
test_data = [(torch.tensor(d, dtype=torch.float32).unsqueeze(0), torch.tensor(l)) for d, l in test_data]
test_dataloader = DataLoader(test_data, shuffle=True, batch_size=4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(1, 16, 8, 4, 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 8, 4, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 8, 4, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(3744, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
    
net = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(10):
    print(f'Epoch {epoch}')
    epoch_loss = 0
    net.train()
    for batch in tqdm(dataloader):
        audio, label = batch
        preds = net(audio)
        loss = loss_fn(preds, label)
        net.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Loss: {epoch_loss / len(dataloader)}')
    net.eval()
    correct = 0
    total = 0
    for batch in test_dataloader:
        audio, label = batch
        preds = net(audio)
        preds = torch.argmax(preds, dim=1)
        correct += (preds == label).sum().item()
        total += len(label)
    print(f'Accuracy: {correct/total*100}%')
