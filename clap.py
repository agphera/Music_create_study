import torch
import laion_clap

# CLAP 모델 로드
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

# 오디오 파일 리스트 (1번부터 10번까지)
original_files = [
    f"original_model_music_{i}.wav" for i in range(1, 11)
]
new_generated_files = [
    f"new_generated_music_{i}.wav" for i in range(1, 11)
]

# 텍스트 프롬프트 리스트 (1번부터 10번까지)
text_prompts = [
    "An energetic EDM track with a fast tempo and powerful bass drops.",
    "A high-energy dance music track featuring rhythmic claps and soaring synths.",
    "A progressive house track with smooth transitions and uplifting melodies.",
    "A driving EDM track with pulsating beats and a dynamic bassline.",
    "A festival-style EDM track with euphoric buildups and explosive drops.",
    "A tropical house track with warm melodies and smooth vocal chops.",
    "An upbeat EDM track with bright pluck synths and bouncy rhythms.",
    "A cinematic EDM track with epic synths and dramatic drum patterns.",
    "A big room house track with anthemic leads and a powerful festival drop.",
    "A melodic techno track with hypnotic arpeggios and deep basslines."
]

# 텍스트 임베딩 생성
text_embed = model.get_text_embedding(text_prompts, use_tensor=True)

# CLAP 점수 계산 및 출력
for i in range(10):
    original_audio_embed = model.get_audio_embedding_from_filelist([original_files[i]], use_tensor=True)
    new_generated_audio_embed = model.get_audio_embedding_from_filelist([new_generated_files[i]], use_tensor=True)

    # 해당 텍스트 프롬프트로 점수 계산
    original_clap_score = torch.matmul(original_audio_embed, text_embed[i].unsqueeze(0).T).item()
    new_generated_clap_score = torch.matmul(new_generated_audio_embed, text_embed[i].unsqueeze(0).T).item()
    score_difference = new_generated_clap_score - original_clap_score

    # 결과 출력
    print(f"Text Prompt: \"{text_prompts[i]}\"")
    print(f"CLAP Score for Original Model - {original_files[i]}: {original_clap_score}")
    print(f"CLAP Score for New Generated Model - {new_generated_files[i]}: {new_generated_clap_score}")
    print(f"Difference (New Generated - Original) for {original_files[i]}: {score_difference}\n")
