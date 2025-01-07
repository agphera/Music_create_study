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

# 입력 텍스트 리스트
prompts = [
    #"A calm and peaceful lo-fi jazz music with gentle Christmas vibes, featuring soft piano and soothing rhythms.",
    #"Warm and cozy Christmas lo-fi music with subtle jingle bells, evoking a positive mood and relaxing holiday night",
    #"Bright lo-fi jazz music with a wintery touch, featuring smooth melodies.",
    #"A quiet and serene lo-fi track perfect for a peaceful winter night, with slow tempos and smooth ambient tones.",
    #"Dreamy and smooth lo-fi Christmas music, with warm tones and subtle background chimes for a relaxed mood."
    #"Relaxing lo-fi music with a calm vibe.",
    #"Cozy Christmas music with soft beats.",
    #"Gentle lo-fi jazz for a quiet evening",
    #"A tranquil lo-fi track with soft piano tones and subtle jingle bells in slow drum beats.",
    #"Smooth lo-fi jazz with warm saxophone melodies and layered guitar riffs in a steady tempo.",
    #"A slow and gentle lo-fi melody featuring light chime effects, soft bass, and relaxing ambient tones",
    #"A warm and nostalgic lo-fi track evoking peaceful winter nights with positive vibes",
    #"Relaxing lo-fi beats with a tranquil and cozy atmosphere for unwinding after a long day",
    #"A dreamy and serene lo-fi melody that brings calmness and a reflective mood.",
    #"A cozy lo-fi track perfect for a snowy winter night by the fireplace with light jingle bells",
    ##"Dreamy lo-fi music for a quiet Christmas Eve, blending warm piano tones and subtle ambient effects",
    #"Relaxing lo-fi beats for a peaceful winter evening under a starlit sky",
    #"A calm and warm lo-fi melody featuring soft piano, subtle jingle bells, and slow drum beats, evoking a cozy Christmas night by the fireplace.",
    #"Dreamy lo-fi jazz with gentle saxophone tones, layered guitar riffs, and ambient chimes, creating a tranquil winter evening mood under the stars",
    #"A relaxing lo-fi track with warm piano melodies, light bell sounds, and steady beats, perfect for reflecting on peaceful winter nights."
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

# 샘플링 설정
max_length = 1024
num_beams = 1
do_sample = True
temperature = 1.0

# 음악 생성 및 저장
for i, input_text in enumerate(prompts):
    # 텍스트 토큰화
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # 음악 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature
        )

    # 오디오 코드를 파일로 저장
    generated_audio = outputs[0]
    sample_rate = 32000
    file_name = f"new_generated_music_{i + 21}.wav"
    torchaudio.save(file_name, generated_audio, sample_rate)
    print(f"Music generation complete. Saved as '{file_name}'")
