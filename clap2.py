import torch
import laion_clap

# CLAP 모델 로드
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

# 오디오 파일 리스트
audio_files = [
    f"original_model_music_{i + 1}.wav" for i in range(5)
] + [
    f"new_generated_music_{i + 1}.wav" for i in range(5)
]

# 동일한 텍스트 프롬프트
text_prompts = [
    "A calm and peaceful lo-fi jazz music with gentle Christmas vibes, featuring soft piano and soothing rhythms.",
    "Warm and cozy Christmas lo-fi music with subtle jingle bells, evoking a positive mood and relaxing holiday night",
    "Bright lo-fi jazz music with a wintery touch, featuring smooth melodies.",
    "A quiet and serene lo-fi track perfect for a peaceful winter night, with slow tempos and smooth ambient tones.",
    "Dreamy and smooth lo-fi Christmas music, with warm tones and subtle background chimes for a relaxed mood."
]

# 텍스트 임베딩 생성
text_embed = model.get_text_embedding(text_prompts, use_tensor=True)

# CLAP 점수 계산 및 출력
for i, audio_file in enumerate(audio_files):
    audio_embed = model.get_audio_embedding_from_filelist([audio_file], use_tensor=True)
    clap_score = torch.matmul(audio_embed, text_embed[i % 5].unsqueeze(0).T).item()
    model_type = "Original" if i < 5 else "P-Tuning"
    print(f"CLAP Score for {model_type} Model - Audio {i % 5 + 1}: {clap_score}")
