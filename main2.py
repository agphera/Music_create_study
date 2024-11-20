import os
import json
import torch
from transformers import (
    AutoProcessor,
    EncodecModel,
    TrainingArguments,
    Trainer,
    MusicgenForConditionalGeneration
)
from peft import PromptEncoderConfig, get_peft_model, TaskType
from datasets import Dataset
import librosa

# warnings.filterwarnings("ignore", category=FutureWarning)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# T5
text_encoder = model.text_encoder

peft_config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=20,
    token_dim=text_encoder.config.hidden_size,
    num_transformer_submodules=1,
    num_attention_heads=text_encoder.config.num_heads,
    num_layers=text_encoder.config.num_layers,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=text_encoder.config.d_ff,
)

text_encoder = get_peft_model(text_encoder, peft_config)

model.text_encoder = text_encoder

for param in model.parameters():
    param.requires_grad = False

for param in model.text_encoder.parameters():
    param.requires_grad = False

for param in model.text_encoder.prompt_encoder.parameters():
    param.requires_grad = True

parameter_size = model.text_encoder.print_trainable_parameters()

audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz").eval()

data_dir = './data'

data = {'text': [], 'audio': []}

for file_name in os.listdir(data_dir):
    if file_name.endswith('.json'):
        json_path = os.path.join(data_dir, file_name)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        description = json_data['description']
        base_name = os.path.splitext(file_name)[0]
        mp3_file = base_name + '.mp3'
        mp3_path = os.path.join(data_dir, mp3_file)
        if os.path.exists(mp3_path):
            data['text'].append(description)
            data['audio'].append(mp3_path)
        else:
            print(f"Warning: mp3 file for {file_name} not found.")

dataset = Dataset.from_dict(data)

max_audio_length = 320000  # 10초 * 32,000Hz

def preprocess_function(example):
    audio_file = example['audio']
    try:
        audio_array, sr = librosa.load(audio_file, sr=32000)
    except Exception as e:
        print(f"Error loading {audio_file}: {e}")
        return None

    audio_tensor = torch.tensor(audio_array).unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, sequence_length)


    if audio_tensor.shape[-1] > max_audio_length:
        audio_tensor = audio_tensor[..., :max_audio_length]
    elif audio_tensor.shape[-1] < max_audio_length:
        pad_size = max_audio_length - audio_tensor.shape[-1]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_size))

    inputs = processor(
        text=[example['text']],
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        encoded_audio = audio_encoder.encode(audio_tensor)

    audio_codes = encoded_audio.audio_codes  # shape: (frames, batch_size, num_codebooks, seq_len)
    frames, batch_size, num_codebooks, seq_len = audio_codes.shape
    labels = audio_codes.reshape(batch_size * num_codebooks, seq_len)

    return {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
        'labels': labels
    }


processed_dataset = dataset.map(preprocess_function, batched=False)
processed_dataset = processed_dataset.filter(lambda x: x is not None)

processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

def data_collator(features):
    if len(features) == 0:
        return None

    return {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'labels': torch.stack([f['labels'] for f in features]),
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir='./musicgen_ptuning_results',
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=10,
    learning_rate=1e-4,
    remove_unused_columns=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
    # tokenizer=processor.tokenizer,  # tokenizer
)

trainer.train()