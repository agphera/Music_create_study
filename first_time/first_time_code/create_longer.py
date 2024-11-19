from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy

# 1. Processor와 Model 불러오기
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

#입력 프롬프트 3가지
text_prompts = [
    "christmas jazz with calm piano chords",
    "Dreamy ambient style of lofi hip-hop music",
    "Christian music with acoustic guitar and sacred atmosphere" #찬송가
]

# 3. 음악 생성 및 저장
for i, prompt in enumerate(text_prompts, start=1):  # i는 1부터 시작
    # 입력 텍스트를 처리
    inputs = processor(
        text=[prompt],  # 각 프롬프트를 하나씩 처리
        padding=True,
        return_tensors="pt",
    )
    
    # 모델을 사용해 음악 생성
    with torch.no_grad():
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=512)
    
    # 샘플링 레이트 가져오기
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    # WAV 파일로 저장 (파일 이름에 번호 추가)
    file_name = f"longer_{i}.wav"
    scipy.io.wavfile.write(file_name, rate=sampling_rate, data=audio_values[0, 0].numpy())
    
    print(f"Music {i} generated from prompt '{prompt}' and saved as {file_name}")
