import comet_ml #로그 및 학습 확인
import torch 
import torchaudio #오디오 파일 처리
import torch.nn as nn
from transformers import (
    T5Tokenizer, #텍스트 -> 토큰화
    MusicgenForConditionalGeneration, #Musicgen조건부생성모델  
    TrainingArguments, #모델 학습 시 필요 설정 정의
    Trainer #설정을 바탕으로 학습 루프 진행
)
from datasets import Dataset, DatasetDict, load_from_disk
import os
import json

tokenizer_name = 't5-base' #t5Tokenizer _ t5-base 모델 사용
model_name = 'facebook/musicgen-small' #뮤직젠 small 모델 사용
prompt_length = 10 #입력 프롬프트 길이
hidden_size = 768 #t5모델의 hidden_size=768. 이에 따라 설정

# Prompt Encoder 초기화
prompt_length = 10
hidden_size = 768
intermediate_size = 1024  # MLP 중간 크기


#모델과 토크나이저 설정
musicgen_model = MusicgenForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
#
musicgen_model.config.decoder.decoder_start_token_id = 0

#가중치 고정 (텍스트 인코더, 디코더, 오디오 인코더)
for param in musicgen_model.text_encoder.parameters():
    param.requires_grad = False #인코더를 학습시키는게 아니라, 프롬프트 벡터만 학습시켜서 삽입하는 것임.
for param in musicgen_model.decoder.parameters():
    param.requires_grad = False
for param in musicgen_model.audio_encoder.parameters():
    param.requires_grad = False

#추가할 프롬프트 벡터 선언
prompt_vector = nn.Parameter(torch.randn(1, prompt_length, hidden_size), requires_grad=True)

# P-Tuning을 위한 추가적인 레이어 (MLP 기반)
class PromptEncoder(nn.Module):
    def __init__(self, prompt_length, hidden_size, intermediate_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, prompt_length, hidden_size), requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, batch_size):
        prompt = self.embedding.expand(batch_size, -1, -1)  # (batch_size, prompt_length, hidden_size)
        prompt = self.mlp(prompt)
        return prompt

prompt_encoder = PromptEncoder(prompt_length, hidden_size, intermediate_size)  # cpu

#입력 텍스트에 프롬프트 공간 추가
def create_inputs_with_prompt(input_texts):
    #입력 텍스트 토큰화
    input_ids = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True).input_ids
    
    #입력 텍스트 앞에 추가할 빈 공간(0) 생성
    prompt_space = torch.zeros((input_ids.size(0), prompt_length), dtype=input_ids.dtype)
    
    #빈 공간 + 토큰화된 입력 => 최종 입력 생성
    return torch.cat([prompt_space, input_ids], dim=1)


#오디오 데이터를 코드북(모델 입력에 사용할 수 있는 벡터 형태)으로 변환
def generate_audio_labels(audio_inputs, sample_rates, target_sample_rate=32000, target_sec=30):
    # (입력 오디오 target_sample_rate=32kHz, 오디오 길이 target_sec=30s 로 설정)
    resampled_audios = []
    target_length = target_sample_rate * target_sec
    #오디오 리샘플링 및 정규화
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
    #오디오 코드 생성
    for i in range(0, audio_input.size(0), 1):
        batch = audio_input[i:i + 1]
        with torch.no_grad():
            audio_encoder_outputs = musicgen_model.audio_encoder(input_values=batch)
            audio_codes_list.append(audio_encoder_outputs.audio_codes)
    merged_audio_codes = torch.cat(audio_codes_list, dim=1)
    return merged_audio_codes[0, ...].permute(0, 2, 1)

#텍스트-오디오 데이터셋 생성
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
        input_texts.append(description) #json 파일에서 입력 텍스트(description 등)를 읽어옴
        waveform, original_sample_rate = torchaudio.load(audio_path)
        audio_inputs.append(waveform) #mp3 파일로부터 오디오 데이터를 로드
        sample_rates.append(original_sample_rate)

    inputs_ids = create_inputs_with_prompt(input_texts) #입력 텍스트에 프롬프트 공간 추가한 것
    labels = generate_audio_labels(audio_inputs, sample_rates) #오디오 코드북(=라벨) 생성
    inputs_ids_list.extend(inputs_ids.numpy())
    labels_list.extend(labels.numpy())

    #데이터셋 생성 (텍스트와 오디오 라벨을 포함한 Dataset객체 생성)
    dataset = Dataset.from_dict({'input_ids': inputs_ids_list, 'labels': labels_list})
    dataset_dict = DatasetDict({'train': dataset})
    dataset_dict.save_to_disk(dataset_dir)


def forward_hook(module, input):
    original_input = input[0]
    batch_size = original_input.size(0)
    
    # P-Tuning: 프롬프트 벡터를 MLP로 처리한 결과를 가져옴
    prompt = prompt_encoder(batch_size)
    
    # 입력 텐서의 앞부분을 프롬프트로 교체
    original_input[:, :prompt_length, :] = prompt
    return (original_input,)



def train_model(dataset_path, output_dir='./musicgen_results', learning_rate=5e-5, batch_size=3, num_epochs=3):
    dataset_dict = load_from_disk(dataset_path)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict.get('eval', None)

    # Forward Hook 등록
    musicgen_model.text_encoder.encoder.block[0].register_forward_pre_hook(forward_hook)
    
    # Prompt Encoder 파라미터 학습을 위한 옵티마이저
    optimizer = torch.optim.Adam(prompt_encoder.parameters(), lr=learning_rate)

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