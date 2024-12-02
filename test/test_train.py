import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import Parameter
from audiocraft.models import MusicGen
from transformers import AutoTokenizer
import wandb
import os
import json

wandb.login()

# Configurations
PROJECT_NAME = 'p-tuning-musicgen'
RUN_NAME = 'ptuning-train'
JSON_PATH = 'data/dataset.json'
CHECKPOINT_DIR = 'checkpoints'
MODEL = 'small'
EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACC_STEPS = 1
LR = 1e-4
PROMPT_LENGTH = 10  # Number of learnable tokens
EMBEDDING_DIM = 768  # Match MusicGen's embedding size

class JSONAudioDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)["data"]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio_path = sample["audio_file"]
        text_condition = f"{sample['description']} Keywords: {', '.join(sample['keywords'])}. Moods: {', '.join(sample['moods'])}."

        # Load audio (convert to tensor or spectrogram if needed)
        audio = self.load_audio(audio_path)

        return audio, text_condition

    def load_audio(self, path):
        # Convert MP3 to tensor, use librosa or torchaudio
        import librosa
        audio, _ = librosa.load(path, sr=32000)  # Match model's expected sample rate
        return torch.tensor(audio)

class PromptConditionProvider(torch.nn.Module):
    def __init__(self, prompt_length, embedding_dim):
        super().__init__()
        self.prompt = Parameter(torch.randn(prompt_length, embedding_dim))  # Learnable prompt tokens

    def forward(self, tokens):
        # Concatenate learnable prompts with token embeddings
        batch_size = tokens.size(0)
        prompt_expanded = self.prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_expanded, tokens], dim=1)
# Initialize MusicGen and tokenizer
model = MusicGen.get_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT2 tokenizer or model-specific tokenizer

# Replace condition_provider with PromptConditionProvider
model.condition_provider = PromptConditionProvider(PROMPT_LENGTH, EMBEDDING_DIM)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


####
# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Dataset and DataLoader
dataset = JSONAudioDataset(JSON_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Loss function
loss_fn = torch.nn.MSELoss()

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, (audio, text) in enumerate(dataloader):
        # Tokenize text and move to device
        tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = tokenized.input_ids.to(device)

        # Forward pass with P-Tuning prompts
        prompts = model.condition_provider(tokens)
        outputs = model(audio.to(device), prompts)

        # Compute loss
        loss = loss_fn(outputs.generated_audio, audio.to(device))
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        if (i + 1) % GRAD_ACC_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Log to wandb
        wandb.log({"epoch": epoch, "loss": loss.item()})

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader)}")

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)
