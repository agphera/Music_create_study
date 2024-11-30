from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import torchaudio

# 학습된 MusicGen 모델과 Processor 로드
processor = AutoProcessor.from_pretrained('./musicgen_ptuning_results/final_model')
trained_model = AutoModelForCausalLM.from_pretrained('./musicgen_ptuning_results/final_model')

# 모델을 평가 모드로 설정
trained_model.eval()

# 텍스트 조건부 입력
test_text = "Christmas Carol jazz with calm and cozy moods in lofi-style"

# 텍스트를 토큰화
inputs = processor(
    text=[test_text],
    padding=True,
    return_tensors="pt"
)

# 텍스트 조건부 음악 생성
with torch.no_grad():
    generated_audio = trained_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=64000,  # MusicGen은 샘플 수로 결정 (여기서는 약 2초)
        temperature=1.0,
        top_k=50
    )

# 생성된 오디오 데이터 후처리 및 저장
generated_audio_waveform = processor.decode(generated_audio)
torchaudio.save("generated_audio.wav", generated_audio_waveform.unsqueeze(0), sample_rate=32000)

print("Audio generated and saved as 'generated_audio.wav'")
