import torch
from torch.utils.data import Dataset
import json
import librosa
from pydub import AudioSegment
import os

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
        # MP3를 WAV로 변환
        if path.endswith(".mp3"):
            wav_path = path.replace(".mp3", ".wav")
            if not os.path.exists(wav_path):
                self.convert_mp3_to_wav(path, wav_path)
            path = wav_path
        try:
            audio, _ = librosa.load(path, sr=32000)
            return torch.tensor(audio)
        except Exception as e:
            print(f"Error loading audio file {path}: {e}")
            return torch.zeros(1)  # 빈 텐서 반환

    def convert_mp3_to_wav(self, mp3_path, wav_path):
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")
            print(f"Converted {mp3_path} to {wav_path}")
        except Exception as e:
            print(f"Error converting {mp3_path} to WAV: {e}")
