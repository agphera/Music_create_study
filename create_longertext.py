from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy

# 1. Processor와 Model 불러오기
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")


inputs = processor(
    text=["A slow-paced jazz Christmas carol at 70 BPM, with a warm and soothing piano melody as the lead. use soft dynamics and rich harmonic progressions, with extended jazz chords like major 7ths and minor 9ths."],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, do_sample=True, guidance_scale=1, max_new_tokens=1024)

#파일 저장
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("longer_t.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())    

print(f"Music generated from longer prompt")
