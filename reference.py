import os
import json
import torch
from transformers import (
    AutoProcessor, #텍스트 -> 토큰화
    EncodecModel, #오디오 파일 -> 코드화
    TrainingArguments, #모델 학습 시 필요한 설정 정의
    Trainer, #TrainingArgs와 모델, 데이터를 불러와 학습 루프 진행, 손실 계산 및 역전파 수행
    MusicgenForConditionalGeneration 
)
from peft import PromptEncoderConfig, get_peft_model, TaskType
from datasets import Dataset
import librosa
import torchaudio

#input_ids와 labels
'''
input_ids
tokenizer 를 이용한 텍스트 입력 토큰화
T5모델의 tokenizer를 이용해 작성됨
현재 json 파일의 description을 인풋 텍스트로 적용함


labels
오디오 파일을 불러와, Encodec의 인코딩 과정을 거친 데이터로 만들어야 함
기존 Musicgen-small 모델의 기본 오디오 인코더를 사용함.
기본 오디오 인코더 모델 : facebook/encodec_32khz

실제 디코더 모델에서 라벨의 형태는,
(배치 사이즈, 시퀸스 길이, 코드북 숫자)의 형태를 필요로 함
'''
self.tokenizer=T5Tokenizer.from_pretrained(tokenizer_name)
input_ids=self.tokenizer(input_texts,return tensors='pt',padding=True, truncation=True).input_ids

####################################################
#labels 생성과정
'''
1. 오디오 파일 불러오기 . (torchaudio 라이브러리 사용)
waveform : sample rate로 쪼개진 오디오 데이터

2. 오디오 파일 통일. 
데이터 샘플링을 기본 오디오 인코더의 설정인 32000으로 맞춤 
이 과정에서 길이가 달라질 수도 있기 때문에, 길이를 맞춤
musicgen-small의 기본 설정은 1채널 오디오를 사용하기에, 채널을 맞춤
'''
#오디오 파일로드
waveform, original_sample_rate=torchaudio.load(audio_path)
#리샘플링 수행
resampled_audio=torchaudio.transforms.Resample(
    orig_freq=original_sample_rate,
    new_freq=32000
)(waveform) #[channels, seq_len]

if audio_input.size(1)==2:
    #2채널 오디오를 1채널로 변환(채널 축 평균
    audio_input=audio_input.mean(dim=1, keepdim=True)

#오디오 길이 통일(패딩 또는 잘라내기)
if resampled_audio.size(1)<target_length:
    pad_amount=target_length-resampled_audio.size(1)
    resampled_audio=torch.nn.functional.pad(resampled_audio,(0,pad_amount), "constant", 0)
elif resampled_audio.size(1)>target_length:
    resampled_audio=resampled_audio[:,:target_length]

'''
3. 오디오 파일 인코딩
불러온 오디오 데이터를 오디오 인코더 모델을 통해 인코딩을 진행함
출력값 : [프레임, 배치 사이즈, 코드북 길이, 시퀸스 길이]
(musicgen 모델은 프레임이 1인 오디오만 이용함)
프레임 : 오디오 입력을 나눠서 처리하기 위한 단위 

4. 데이터 맞춤
musicgen 디코더 모델에서는 다음과 같은 labels 형태를 필요로 함
[배치 사이즈, 시퀸스 길이, 코드북 길이]
출력 형태를 조절해 같은 형태로 수정함
'''
audio_encoder_outputs=self.musicgen_model.audio_encoder(input_values=batch)
audio_codes=audio_encoder_outputs.audio_codes #[frames, batch_size, codebooks, seq_len]
#[frames, batch_size, codebooks, seq_len]->[전체 batch_size, seq_len,codebooks]
labels=merged_audio_codes[0, ...].permute(0,2,1)

##################################################

#모델 불러오기
class MusicgenPromptTuner:
    def __init__(self, tokenizer_name='T5-base',model_name='facebook/musicgen-small',prompt_length=10, hidden_size=768):
        self.musicgen_model=MusicgenForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer=T5Tokenizer.from_pretrained(tokenizer_name)
        self.musicgen_model.config.decoder.decoder_start_toke_id=0

#가중치 고정

for param in self.musicgen_mode.text_encoder.parameters():
    param.requires_grad = False

for param in self.musicgen_model.decoder.parameters():
    param.requires_grad = False

for param in self.musicgen_model.audio_encoder.parameters():
    param.requires_grad = False

'''
프롬프트 벡터 전처리 (*Data Preprocessing : input_ids)

프롬프트 튜닝 훈련 시, 텍스트 입력 앞에 프롬프트 벡터를 추가해야 함
데이터 셋 생성 시, 원활한 훈련을 위해 토큰화된 텍스트 앞 부분에 공간에 임시 텐서를 생성해 추가함
처리 결과 : [ 추가할 프롬프트 길이의 텐서 + 토큰화된 입력 텍스트 텐서 ]

'''
#프롬프트 벡터 전처리
input_ids=self.tokenizer(input_texts, return_tensors='pt',padding=True, truncation=True).input_ids
#프롬프트 벡터를 위한 빈 공간을 추가한 새로운 텐서 생성
prompt_space=torch.zeros((batch_size, prompt_length),dtype=input_ids.dtype)

#프롬프트 공간과 입력 시퀀스를 결합하여 최종 입력 텐서를 생성
input_with_prompt=torch.cat([prompt_space,input_ids],dim=1)


#프롬프트 벡터 생성 및 옵티마이저 등록
self.prompt_vector=nn.Parameter(torch.randn(1,self.prompt_length, self.hidden_size), requires_grad=True)
optimizer=torch.optim.Adam([self.prompt_vector],lr=learning_rate)

'''
forward hook : 모델의 forward 과정에서 내부 특정 레이어에 로직을 추가할 수 있는 함수

데이터 셋 전처리 과정에서 input_ids 앞에 생성해 둔 임시 텐서 공간에,
생성한 프롬프트 벡터를 추가하는 로직을 작성
이를 내부 인코더 첫 번째 레이어에 등록해 훈련 시, 프롬프트 튜닝을 진행할 수 있도록 함

'''
def forward_hook(self, module, input):
    #input은 튜플형태
    original_input=input[0]
    batch_size, _, hidden_size=original_input.shape

    #프롬프트 길이 계산
    prompt_length=self.prompt_length

    #학습 가능한 프롬프트 벡터를 배치 크기에 맞게 확장
    expanded_prompt=self.prompt_vector.to(original_input.device).expand(batch_size,prompt_length,hidden_size)

    #빈 공간(프롬프트 부분)을 학습 가능한 프롬프트 벡터로 대체
    original_input[:, :prompt_length, :]=expanded_prompt
    
    #출력 상태를 그대로 반환
    return (original_input,)
#forward hook을 인코더의 첫번째 레이어에 등록
self.musicgen_model.text_encoder.encoder.block[0].register_forward_pre_hook(self.forward_hook)


# Training Arguments
training_args = TrainingArguments(
    output_dir='./musicgen_ptuning_results',
    #per_device_train_batch_size=1,
    
    num_train_epochs=3,
    logging_steps=10,
    save_steps=10,
    learning_rate=1e-4,
    #remove_unused_columns=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
    # tokenizer=processor.tokenizer,  # tokenizer
)