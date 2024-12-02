from transformers import T5Tokenizer, MusicgenForConditionalGeneration
import torch
import torchaudio

# 모델 및 토크나이저 경로 설정
model_dir = './ptuning'  # 학습된 모델이 저장된 디렉토리
tokenizer_name = 't5-base'

# 모델과 토크나이저 로드
model = MusicgenForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
model.eval()

# 입력 텍스트
input_text = "calm and cozy christmas jazz with piano and lofi style"

# 텍스트 토큰화
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# 음악 생성
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_length=1024,
        num_beams=1,
        do_sample=True,
        temperature=1.0
    )

# 오디오 코드를 파일로 저장
generated_audio = outputs[0]
sample_rate = 32000
torchaudio.save("ptuned_gen.wav", generated_audio, sample_rate)

print("Music generation complete. Saved as 'generated_music.wav'")
