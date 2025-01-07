import torch
import laion_clap

# CLAP 모델 로드
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

# 오디오 파일 리스트 (21번부터 30번까지)
original_files = [
    f"original_model_music_{i}.wav" for i in range(21, 31)
]
new_generated_files = [
    f"new_generated_music_{i}.wav" for i in range(21, 31)
]

# 동일한 텍스트 프롬프트
text_prompts = [
    "A dreamy lo-fi jazz melody blending soft piano, smooth saxophone, and warm guitar riffs, evoking the cozy ambiance of a snowy winter night with subtle jingle bells in the background.",
    "A tranquil lo-fi track with layered ambient tones, featuring a calm winter vibe with steady drum beats, relaxing chime effects, and gentle Christmas melodies.",
    "A warm and nostalgic lo-fi jazz composition with soft piano chords, serene saxophone solos, and light electronic textures, capturing the essence of peaceful Christmas evenings by the fireplace.",
    "An uplifting lo-fi beat enriched with vibrant saxophone melodies, layered guitar harmonics, and soothing ambient tones, perfect for a joyful yet serene Christmas morning.",
    "A vintage-inspired lo-fi track with slow tempos and mellow drum patterns, blending nostalgic piano tones, smooth jazz elements, and subtle sleigh bells to evoke warm winter memories.",
    "A soothing lo-fi jazz melody featuring soft guitar riffs, layered chime effects, and tranquil piano notes, evoking a peaceful and reflective mood for a starlit Christmas Eve.",
    "A positive lo-fi track with layered textures of warm saxophone, dreamy piano melodies, and steady drum beats, capturing the joyful essence of a snowy winter landscape under the moonlight.",
    "A serene lo-fi composition combining light ambient wind chimes, nostalgic piano harmonies, and gentle percussion, creating the perfect mood for a relaxing winter evening.",
    "An intricate lo-fi jazz arrangement with overlapping melodies of soft saxophone, gentle guitar riffs, and subtle electronic sounds, evoking a calm yet vibrant Christmas atmosphere.",
    "A mellow lo-fi tune enriched with tranquil piano chords, warm ambient textures, and faint jingle bell chimes, designed to bring comfort and peace during a quiet winter night."
]

# 텍스트 임베딩 생성
text_embed = model.get_text_embedding(text_prompts, use_tensor=True)

# CLAP 점수 계산 및 출력
for i, audio_number in enumerate(range(21, 31)):
    original_audio_embed = model.get_audio_embedding_from_filelist([original_files[i]], use_tensor=True)
    new_generated_audio_embed = model.get_audio_embedding_from_filelist([new_generated_files[i]], use_tensor=True)

    # 각각의 텍스트 프롬프트에 대해 점수 계산
    original_clap_score = torch.matmul(original_audio_embed, text_embed[i].unsqueeze(0).T).item()
    new_generated_clap_score = torch.matmul(new_generated_audio_embed, text_embed[i].unsqueeze(0).T).item()

    print(f"CLAP Score for Original Model - Audio {audio_number}: {original_clap_score}")
    print(f"CLAP Score for P-Tuning Model - Audio {audio_number}: {new_generated_clap_score}")