from transformers import T5Tokenizer, MusicgenForConditionalGeneration
import torch
import torchaudio

# 모델 및 토크나이저 경로 설정
model_dir = './ptuning_250'  # 학습된 모델이 저장된 디렉토리
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
    #"A dreamy lo-fi jazz melody blending soft piano, smooth saxophone, and warm guitar riffs, evoking the cozy ambiance of a snowy winter night with subtle jingle bells in the background.",
    #"A tranquil lo-fi track with layered ambient tones, featuring a calm winter vibe with steady drum beats, relaxing chime effects, and gentle Christmas melodies.",
    #"A warm and nostalgic lo-fi jazz composition with soft piano chords, serene saxophone solos, and light electronic textures, capturing the essence of peaceful Christmas evenings by the fireplace.",
    ##"An uplifting lo-fi beat enriched with vibrant saxophone melodies, layered guitar harmonics, and soothing ambient tones, perfect for a joyful yet serene Christmas morning.",
    #"A vintage-inspired lo-fi track with slow tempos and mellow drum patterns, blending nostalgic piano tones, smooth jazz elements, and subtle sleigh bells to evoke warm winter memories.",
    #"A soothing lo-fi jazz melody featuring soft guitar riffs, layered chime effects, and tranquil piano notes, evoking a peaceful and reflective mood for a starlit Christmas Eve.",
    #"A positive lo-fi track with layered textures of warm saxophone, dreamy piano melodies, and steady drum beats, capturing the joyful essence of a snowy winter landscape under the moonlight.",
    #"A serene lo-fi composition combining light ambient wind chimes, nostalgic piano harmonies, and gentle percussion, creating the perfect mood for a relaxing winter evening.",
    #"An intricate lo-fi jazz arrangement with overlapping melodies of soft saxophone, gentle guitar riffs, and subtle electronic sounds, evoking a calm yet vibrant Christmas atmosphere.",
    #"A mellow lo-fi tune enriched with tranquil piano chords, warm ambient textures, and faint jingle bell chimes, designed to bring comfort and peace during a quiet winter night."
    #"An energetic EDM track with a fast tempo, powerful bass drops, and uplifting synth melodies for a nightclub vibe.",
#"High-energy dance music featuring rhythmic claps, deep kicks, and soaring synth leads for a festival anthem.",
#"A progressive house track with smooth transitions, atmospheric pads, and a euphoric buildup to a massive drop.",
#"A club-ready EDM track with pulsating beats, crisp hi-hats, and a catchy synth hook.",
#"A bass-heavy trap-style EDM track with snappy snares, punchy basslines, and an aggressive drop.",
#"A melodic dubstep track with wobbly bass, haunting vocal chops, and uplifting chord progressions.",
#"A fast-paced techno beat with relentless kicks, metallic percussion, and hypnotic synth arpeggios.",
#"A tropical house track with chilled beats, smooth vocal samples, and playful marimba melodies.",
#"An upbeat electronic track with funky basslines, bright plucks, and a joyful atmosphere perfect for a summer party.",
#"A high-intensity big room house track with explosive drops, anthemic melodies, and festival-style energy."
#"A high-energy EDM track with a pounding bassline, crisp snares, and a euphoric buildup leading to an explosive drop.",
#"A progressive house track featuring atmospheric synth pads, uplifting melodies, and a seamless transition into a powerful bass drop.",
#"A festival-style big room house track with anthemic chord progressions, driving kicks, and a thrilling, crowd-pleasing drop.",
#"A melodic techno track with hypnotic arpeggios, pulsating basslines, and an industrial atmosphere perfect for late-night raves.",
#"A tropical house track with breezy plucks, smooth vocal chops, and a light, sunny vibe ideal for beach parties.",
#"An energetic trap-EDM hybrid track with hard-hitting 808s, rapid hi-hats, and a dynamic switch-up at the drop.",
#"A dubstep track with wobbly bass, sharp synth stabs, and a menacing drop designed to energize the dancefloor.",
#"A future bass track with emotional chord progressions, crisp percussions, and soaring vocal chops for a feel-good atmosphere.",
#"A psytrance track with fast-paced basslines, ethereal synth leads, and a hypnotic rhythm to immerse listeners in a psychedelic experience.",
#"An electro house track with gritty basslines, punchy drums, and a funky groove for a high-energy club environment."
#"A chillstep track with smooth basslines, soft piano chords, and atmospheric pads for a relaxed yet engaging vibe.",
#"An experimental EDM track with glitchy effects, syncopated rhythms, and unpredictable drop transitions.",
#"A festival anthem with energetic buildups, layered synth harmonies, and a massive stadium-filling drop.",
#"A deep house track with groovy basslines, subtle vocal cuts, and a laid-back yet driving rhythm.",
#"A future house track featuring bouncy bass, sharp pluck synths, and a catchy melodic hook.",
#"A cinematic electronic track with orchestral strings, epic percussion, and a dynamic, uplifting climax.",
#"A hardstyle track with pounding kicks, distorted basslines, and aggressive synth leads for high-energy intensity.",
#"A drum and bass track with fast breakbeats, smooth sub-bass, and jazzy chord progressions for a sophisticated edge.",
#"A hybrid EDM track combining orchestral elements with energetic electronic drops, perfect for dramatic moments.",
#"A groove-heavy tech house track with punchy drums, rolling basslines, and minimal synth stabs for a club-ready vibe."
        #"Dreamy and cozy Christmas jazz lofi track with a relaxing piano melody.",
    #"A calm and peaceful Christmas lofi track with warm vibes and subtle saxophone sounds.",
    ##"Slow-tempo lofi music with a cozy winter feeling and soft jingle bells.",
    #"A relaxing jazz lofi track with a positive Christmas mood and piano melody.",
    ##"A peaceful Christmas lofi track with subtle electronic sounds and calm winter vibes.",
    #"Warm and tranquil lofi music featuring smooth jazz elements and subtle holiday bells.",
    #"A mellow and serene lofi track with a cozy Christmas atmosphere.",
    #"Relaxing lofi music with soft piano and a soothing winter ambiance.",
    ##"A dreamy lofi jazz track with a slow beat and subtle festive elements.",
    #"A warm and nostalgic lofi track perfect for a calm Christmas evening.",
    #"Cozy and gentle lofi jazz music with a relaxing piano and saxophone melody.",
    #"A peaceful Christmas carol lofi track with a warm and calm mood.",
    #"A positive and dreamy lofi jazz track with subtle holiday jingles.",
    #"A soothing lofi track with a slow tempo and serene winter vibes.",
    #"A tranquil lofi Christmas jazz music featuring smooth piano and relaxing beats.",
    #"A nostalgic Christmas lofi track with warm and cozy melodies for peaceful nights.",
    #"A serene and positive lofi jazz track with subtle holiday bells and a cozy atmosphere.",
    #"A calming lofi track with a dreamy winter vibe and light electronic sounds.",
    ##"Soft and warm lofi music with a slow beat and a relaxing holiday mood.",
    #"A calm Christmas lofi jazz track with piano, saxophone, and a serene ambiance.",
    #"Relaxing lofi music with gentle beats and a tranquil winter feeling.",
    ##"A smooth Christmas lofi track with subtle holiday sounds and a cozy mood.",
    #"A tranquil lofi jazz carol with a dreamy winter atmosphere and relaxing melody.",
    #"A peaceful Christmas jazz lofi track with subtle electronic elements and calm vibes.",
    #"A nostalgic and soothing lofi track perfect for a quiet Christmas night.",
    #"A warm and positive lofi track with a serene Christmas atmosphere.",
    #"A mellow lofi jazz track with cozy vibes and a relaxing winter melody.",
    #"A slow-tempo lofi track with soft piano and subtle festive jingles.",
    #"A dreamy and tranquil lofi Christmas jazz track with relaxing saxophone.",
    #"A relaxing winter lofi track with warm tones and a calm holiday ambiance."
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
    file_name = f"new_generated_music_{i + 91}.wav"
    torchaudio.save(file_name, generated_audio, sample_rate)
    print(f"Music generation complete. Saved as '{file_name}'")
