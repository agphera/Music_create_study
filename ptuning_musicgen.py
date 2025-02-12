import comet_ml  # Log and monitor training
import torch
import torchaudio  # Handle audio files
import torch.nn as nn
from transformers import (
    T5Tokenizer,  # Tokenize text
    MusicgenForConditionalGeneration,  # Musicgen conditional generation model
    TrainingArguments,  # Define required settings for model training
    Trainer,  # Execute training loop with the defined settings
)
from datasets import Dataset, DatasetDict, load_from_disk
import os
import json
from sklearn.model_selection import train_test_split

# Configure the model and tokenizer
tokenizer_name = "t5-base"  # Use t5-base model for T5Tokenizer
model_name = "facebook/musicgen-small"  # Use Musicgen small model
prompt_length = 10  # Input prompt length
hidden_size = 768  # hidden_size of the t5 model is 768
intermediate_size = 1024  # ptuning: Intermediate size for MLP

musicgen_model = MusicgenForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
# Set the decoder start token (required for Musicgen)
musicgen_model.config.decoder.decoder_start_token_id = 0

# Freeze weights (text encoder, decoder, audio encoder)
for param in musicgen_model.text_encoder.parameters():
    param.requires_grad = False
for param in musicgen_model.decoder.parameters():
    param.requires_grad = False
for param in musicgen_model.audio_encoder.parameters():
    param.requires_grad = False

# Additional layer for P-Tuning (MLP-based)
class PromptEncoder(nn.Module):
    """
    Define a learnable prompt embedding and convert it with MLP to integrate into the model
    """
    def __init__(self, prompt_length, hidden_size, intermediate_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, prompt_length, hidden_size), requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, batch_size):
        """
        Expand learnable prompt embedding to match the batch size and transform it with MLP
        """
        prompt = self.embedding.expand(batch_size, -1, -1)  # (batch_size, prompt_length, hidden_size)
        prompt = self.mlp(prompt)  # Transform with MLP
        return prompt

# Initialize the prompt encoder
prompt_encoder = PromptEncoder(prompt_length, hidden_size, intermediate_size)

# Add prompt space to input text
def create_inputs_with_prompt(input_texts):
    # Tokenize the input text
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids

    # Create empty space (0) to add in front of the input text
    prompt_space = torch.zeros((input_ids.size(0), prompt_length), dtype=input_ids.dtype)

    # Combine the empty space with tokenized input to create the final input
    return torch.cat([prompt_space, input_ids], dim=1)

# Convert audio data to codebooks (vector format for model input)
def generate_audio_labels(audio_inputs, sample_rates, target_sample_rate=32000, target_sec=30):
    resampled_audios = []
    target_length = target_sample_rate * target_sec
    # Resample and normalize audio
    for waveform, original_sample_rate in zip(audio_inputs, sample_rates):
        resampled_audio = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=32000)(waveform)
        if resampled_audio.size(1) < target_length:
            pad_amount = target_length - resampled_audio.size(1)
            resampled_audio = torch.nn.functional.pad(resampled_audio, (0, pad_amount), "constant", 0)
        elif resampled_audio.size(1) > target_length:
            resampled_audio = resampled_audio[:, :target_length]
        resampled_audios.append(resampled_audio)

    audio_input = torch.stack(resampled_audios)
    if audio_input.size(1) == 2:
        audio_input = audio_input.mean(dim=1, keepdim=True)
    # Generate audio codes
    audio_codes_list = []
    for i in range(0, audio_input.size(0), 1):
        batch = audio_input[i : i + 1]
        with torch.no_grad():
            audio_encoder_outputs = musicgen_model.audio_encoder(input_values=batch)
            audio_codes_list.append(audio_encoder_outputs.audio_codes)
    merged_audio_codes = torch.cat(audio_codes_list, dim=1)
    return merged_audio_codes[0, ...].permute(0, 2, 1)

# Create text-audio dataset
def create_musicgen_dataset(data_dir, dataset_dir):
    input_texts = []
    audio_inputs = []
    sample_rates = []
    file_pairs = []
    for f in os.listdir(data_dir):
        if f.endswith(".json"):
            json_path = os.path.join(data_dir, f)
            mp3_path = os.path.join(data_dir, f.replace(".json", ".mp3"))
            wav_path = os.path.join(data_dir, f.replace(".json", ".wav"))
            
            if os.path.exists(mp3_path):
                file_pairs.append((json_path, mp3_path))
            if os.path.exists(wav_path):
                file_pairs.append((json_path, wav_path))

    for json_path, audio_path in file_pairs:
        with open(json_path, "r") as json_file:
            data = json.load(json_file)
            description = data.get("description", "")
            input_texts.append(description)

        waveform, original_sample_rate = torchaudio.load(audio_path)
        audio_inputs.append(waveform)
        sample_rates.append(original_sample_rate)

    inputs_ids = create_inputs_with_prompt(input_texts)  # Add prompt space to input text
    labels = generate_audio_labels(audio_inputs, sample_rates)  # Generate audio codebooks (labels)

    # Split dataset
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        inputs_ids.numpy(), labels.numpy(), test_size=0.2, random_state=42
    )
    train_dataset = Dataset.from_dict({"input_ids": train_texts, "labels": train_labels})
    eval_dataset = Dataset.from_dict({"input_ids": eval_texts, "labels": eval_labels})
    dataset_dict = DatasetDict({"train": train_dataset, "eval": eval_dataset})
    
    # Save dataset
    dataset_dict.save_to_disk(dataset_dir)

# When the model's forward function is called, insert prompt embeddings at the beginning of input text
# Expand the original input text tensor and include prompt embeddings
def forward_hook(module, input):
    original_input = input[0]
    batch_size = original_input.size(0)
    prompt = prompt_encoder(batch_size)
    original_input[:, :prompt_length, :] = prompt
    return (original_input,)

# Train the model
def train_model(dataset_path, output_dir="./musicgen_results", learning_rate=5e-5, batch_size=3, num_epochs=20):
    dataset_dict = load_from_disk(dataset_path)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["eval"]

    musicgen_model.text_encoder.encoder.block[0].register_forward_pre_hook(forward_hook)

    optimizer = torch.optim.Adam(prompt_encoder.parameters(), lr=learning_rate)

    training_args = TrainingArguments(
        logging_steps=1,
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["comet_ml"],
    )

    trainer = Trainer(
        model=musicgen_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None),
    )
    trainer.train()
    musicgen_model.save_pretrained(output_dir)


# Execution
data_path = "./data/data"
dataset_path = "./data/dataset"

create_musicgen_dataset(data_path, dataset_path)
train_model(
    dataset_path, 
    output_dir="./ptuning", 
    learning_rate= 3e-5,
    batch_size=4,
    num_epochs=50)
