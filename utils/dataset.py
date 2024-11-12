# utils/dataset.py

import torch
from torch.utils.data import Dataset
import json
import librosa

class JSONAudioDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)["data"]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio_path = sample["audio_file"]
        text_condition = f"{sample['description']} Keywords: {', '.join(sample['keywords'])}. Moods: {', '.join(sample['moods'])}."
        audio = self.load_audio(audio_path)
        return audio, text_condition

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=32000)
        return torch.tensor(audio)
