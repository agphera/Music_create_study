# Import MusicGen and tokenizer
from audiocraft.models import MusicGen
from transformers import AutoTokenizer

# Import custom modules
from models.peft_provider import PEFTPConditionProvider
from utils.dataset import JSONAudioDataset
from models.train_peft import train_model
from generate import generate_music

import torch
from torch.utils.data import DataLoader

import os 

# Configuration
#JSON_PATH = "data/Silent-Night.json"
#JSON_PATH = "data"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "data", "Silent-Night.json")
CHECKPOINT_DIR = "checkpoints"
EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACC_STEPS = 1
LR = 1e-4
PROMPT_LENGTH = 10
EMBEDDING_DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MusicGen and tokenizer
model = MusicGen.get_pretrained("small")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add pad_token for GPT-2 tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Use PEFT-based PromptConditionProvider
prompt_provider = PEFTPConditionProvider(
    prompt_length=10,
    hidden_size = 768,  # MusicGen 모델의 hidden_size 설정
    num_transformer_submodules = 12,  # Transformer 서브모듈 수 설정
    num_attention_heads = 12,  # Attention 헤드 수
    num_layers = 12,  # Transformer 레이어 수
)
model.condition_provider = prompt_provider

# Move model components to the specified device
model.lm = model.lm.to(DEVICE)  # 내부 언어 모델 이동
if hasattr(model, "condition_provider"):
    model.condition_provider = model.condition_provider.to(DEVICE)

# Prepare dataset and dataloader
dataset = JSONAudioDataset(JSON_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train the model
train_model(model, tokenizer, dataloader, DEVICE, EPOCHS, GRAD_ACC_STEPS, LR, CHECKPOINT_DIR)

# Generate music
text_condition = "A warm and cozy winter melody with lofi jazz vibes."
generate_music(model, tokenizer, text_condition, DEVICE, "generated_music.wav")
