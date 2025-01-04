from transformers import T5Tokenizer, MusicgenForConditionalGeneration
import torch
import torchaudio

# 원본 MusicGen 모델 및 토크나이저 경로
model_name = "facebook/musicgen-small"  # 원본 모델 이름
tokenizer_name = "t5-base"

# 모델 및 토크나이저 로드
model = MusicgenForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
model.eval()

# 텍스트 프롬프트
prompts = [
    "A calm and peaceful lo-fi jazz music with gentle Christmas vibes, featuring soft piano and soothing rhythms.",
    "Warm and cozy Christmas lo-fi music with subtle jingle bells, evoking a positive mood and relaxing holiday night",
    "Bright lo-fi jazz music with a wintery touch, featuring smooth melodies.",
    "A quiet and serene lo-fi track perfect for a peaceful winter night, with slow tempos and smooth ambient tones.",
    "Dreamy and smooth lo-fi Christmas music, with warm tones and subtle background chimes for a relaxed mood."
]

# 오디오 생성 설정
sample_rate = 32000
max_length = 1024
num_beams = 1
do_sample = True
temperature = 1.0

# 오디오 생성 및 저장
for i, input_text in enumerate(prompts):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature
        )
    
    # 오디오 저장
    generated_audio = outputs[0]
    file_name = f"original_model_music_{i + 1}.wav"
    torchaudio.save(file_name, generated_audio, sample_rate)
    print(f"Generated and saved: {file_name}")
