import comet_ml
import torch
import torchaudio
import torch.nn as nn
from transformers import T5Tokenizer, MusicgenForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_from_disk
import os
import json
#os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

tokenizer_name = 't5-base'
model_name = 'facebook/musicgen-small'
prompt_length = 10
hidden_size = 768

musicgen_model = MusicgenForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
musicgen_model.config.decoder.decoder_start_token_id = 0

for param in musicgen_model.text_encoder.parameters():
    param.requires_grad = False
for param in musicgen_model.decoder.parameters():
    param.requires_grad = False
for param in musicgen_model.audio_encoder.parameters():
    param.requires_grad = False

prompt_vector = nn.Parameter(torch.randn(1, prompt_length, hidden_size), requires_grad=True)


def create_inputs_with_prompt(input_texts):
    input_ids = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).input_ids
    prompt_space = torch.zeros((input_ids.size(0), prompt_length), dtype=input_ids.dtype)
    return torch.cat([prompt_space, input_ids], dim=1)


def generate_audio_labels(audio_inputs, sample_rates, target_sample_rate=32000, target_sec=30):
    resampled_audios = []
    target_length = target_sample_rate * target_sec
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

    audio_codes_list = []
    for i in range(0, audio_input.size(0), 1):
        batch = audio_input[i:i + 1]
        with torch.no_grad():
            audio_encoder_outputs = musicgen_model.audio_encoder(input_values=batch)
            audio_codes_list.append(audio_encoder_outputs.audio_codes)
    merged_audio_codes = torch.cat(audio_codes_list, dim=1)
    return merged_audio_codes[0, ...].permute(0, 2, 1)


def create_musicgen_dataset(data_dir, dataset_dir):
    inputs_ids_list = []
    labels_list = []
    input_texts = []
    audio_inputs = []
    sample_rates = []
    file_pairs = [
        (os.path.join(data_dir, f), os.path.join(data_dir, f.replace('.json', '.mp3')))
        for f in os.listdir(data_dir) if f.endswith('.json')
    ]
    for json_path, audio_path in file_pairs:
        with open(json_path, 'r') as json_file:
            description = json.load(json_file)['description']
        input_texts.append(description)
        waveform, original_sample_rate = torchaudio.load(audio_path)
        audio_inputs.append(waveform)
        sample_rates.append(original_sample_rate)

    inputs_ids = create_inputs_with_prompt(input_texts)
    labels = generate_audio_labels(audio_inputs, sample_rates)
    inputs_ids_list.extend(inputs_ids.numpy())
    labels_list.extend(labels.numpy())

    dataset = Dataset.from_dict({'input_ids': inputs_ids_list, 'labels': labels_list})
    dataset_dict = DatasetDict({'train': dataset})
    dataset_dict.save_to_disk(dataset_dir)


def forward_hook(module, input):
    original_input = input[0]
    expanded_prompt = prompt_vector.to(original_input).expand(original_input.size(0), prompt_length, hidden_size)
    original_input[:, :prompt_length, :] = expanded_prompt
    return (original_input,)


def train_model(dataset_path, output_dir='./musicgen_results', learning_rate=5e-5, batch_size=3, num_epochs=3):
    dataset_dict = load_from_disk(dataset_path)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict.get('eval', None)

    musicgen_model.text_encoder.encoder.block[0].register_forward_pre_hook(forward_hook)
    optimizer = torch.optim.Adam([prompt_vector], lr=learning_rate)

    training_args = TrainingArguments(
        logging_steps=1,
        output_dir=output_dir,
        evaluation_strategy="epoch" if eval_dataset else "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        report_to=["comet_ml"]
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


# 실행
data_path = "./data/data"
dataset_path = "./data/dataset"

create_musicgen_dataset(data_path, dataset_path)
train_model(dataset_path, output_dir='./ptuning', learning_rate=5e-5, batch_size=1, num_epochs=5)