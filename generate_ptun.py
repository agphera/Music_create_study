from transformers import T5Tokenizer, MusicgenForConditionalGeneration
import torch
import torchaudio


model_dir = './ptuning'  # directory of saved p-tuning model
tokenizer_name = 't5-base'


model = MusicgenForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
model.eval()

# text input
prompts = [
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

max_length = 1024
num_beams = 1
do_sample = True
temperature = 1.0

#generate & save audio files
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

    generated_audio = outputs[0]
    sample_rate = 32000
    file_name = f"new_generated_music_{i + 1}.wav" 
    torchaudio.save(file_name, generated_audio, sample_rate)
    print(f"Music generation complete. Saved as '{file_name}'")
