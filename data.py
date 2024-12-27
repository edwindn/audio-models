import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.models import wav2vec2_base
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#git clone https://github.com/PerceptiLabs/yesno_voice_recognition

wav2vec = wav2vec2_base().to(device)
processor = wav2vec.feature_extractor
encoder = wav2vec.encoder

def load_wav_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

yes_data = []
no_data = []
sr = 44100

lens = []

print('Extracting audio files...')
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
fixed_len = 60000
for i, (d, l) in enumerate(test_data):
    if d.shape[0] < fixed_len:
        d = np.pad(d, (0, fixed_len - d.shape[0]), 'constant')
    else:
        d = d[:fixed_len]
    test_data[i] = (d, l)

data = [(torch.tensor(d, dtype=torch.float32).unsqueeze(0), torch.tensor(l)) for d, l in data]
test_data = [(torch.tensor(d, dtype=torch.float32).unsqueeze(0), torch.tensor(l)) for d, l in test_data]

print('Encoding files...')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
wav2vec_train = []
wav2vec_test = []
for d in tqdm(data):
    torch.cuda.empty_cache()
    audio, label = d
    audio = audio.to(device)
    audio = processor(audio, length=fixed_len)[0]
    audio = encoder(audio)
    wav2vec_train.append((audio, label))
for d in tqdm(test_data):
    torch.cuda.empty_cache()
    audio, label = d
    audio = audio.to(device)
    audio = processor(audio, length=fixed_len)[0]
    audio = encoder(audio)
    wav2vec_test.append((audio, label))

dataloader = DataLoader(wav2vec_train, shuffle=True, batch_size=4)
test_dataloader = DataLoader(wav2vec_test, shuffle=True, batch_size=4)

torch.save(dataloader, 'data/train_loader_wav2vec.pth')
torch.save(test_dataloader, 'data/test_loader_wav2vec.pth')
print('Files saved successfully.')
