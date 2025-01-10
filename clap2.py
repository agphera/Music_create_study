import torch
import laion_clap

# CLAP 모델 로드
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

# 오디오 파일 리스트 (21번부터 30번까지)
original_files = [
    f"original_model_music_{i}.wav" for i in range(91, 101)
]
new_generated_files = [
    f"new_generated_music_{i}.wav" for i in range(91,101)
]

# 동일한 텍스트 프롬프트
text_prompts = [
#"An energetic EDM track with a fast tempo, powerful bass drops, and uplifting synth melodies for a nightclub vibe.",
#"High-energy dance music featuring rhythmic claps, deep kicks, and soaring synth leads for a festival anthem.",
##"A progressive house track with smooth transitions, atmospheric pads, and a euphoric buildup to a massive drop.",
#"A club-ready EDM track with pulsating beats, crisp hi-hats, and a catchy synth hook.",
#"A bass-heavy trap-style EDM track with snappy snares, punchy basslines, and an aggressive drop.",
#"A melodic dubstep track with wobbly bass, haunting vocal chops, and uplifting chord progressions.",
#"A fast-paced techno beat with relentless kicks, metallic percussion, and hypnotic synth arpeggios.",
#"A tropical house track with chilled beats, smooth vocal samples, and playful marimba melodies.",
#"An upbeat electronic track with funky basslines, bright plucks, and a joyful atmosphere perfect for a summer party.",
#"A high-intensity big room house track with explosive drops, anthemic melodies, and festival-style energy."
#"A high-energy EDM track with a pounding bassline, crisp snares, and a euphoric buildup leading to an explosive drop.",
#"A progressive house track featuring atmospheric synth pads, uplifting melodies, and a seamless transition into a powerful bass drop.",
##"A festival-style big room house track with anthemic chord progressions, driving kicks, and a thrilling, crowd-pleasing drop.",
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

# 텍스트 임베딩 생성
text_embed = model.get_text_embedding(text_prompts, use_tensor=True)

# CLAP 점수 계산 및 출력
for i, audio_number in enumerate(range(91,101)):
    original_audio_embed = model.get_audio_embedding_from_filelist([original_files[i]], use_tensor=True)
    new_generated_audio_embed = model.get_audio_embedding_from_filelist([new_generated_files[i]], use_tensor=True)

    # 각각의 텍스트 프롬프트에 대해 점수 계산
    original_clap_score = torch.matmul(original_audio_embed, text_embed[i].unsqueeze(0).T).item()
    new_generated_clap_score = torch.matmul(new_generated_audio_embed, text_embed[i].unsqueeze(0).T).item()
    score_difference = new_generated_clap_score - original_clap_score

    print(f"CLAP Score for Original Model - Audio {audio_number}: {original_clap_score}")
    print(f"CLAP Score for P-Tuning Model - Audio {audio_number}: {new_generated_clap_score}")
    print(f"Difference (P-Tuning - Original) for Audio {audio_number}: {score_difference}")