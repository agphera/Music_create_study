import torch
import soundfile as sf

def generate_music(model, tokenizer, text_condition, device, output_path):
    """
    Generate music from the trained MusicGen model.

    Args:
        model: The trained MusicGen model.
        tokenizer: Tokenizer for processing text conditions.
        text_condition (str): Text-based condition for generating music.
        device: The device (CPU or GPU) for inference.
        output_path (str): Path to save the generated audio.

    Returns:
        None. Saves the generated audio to the specified file path.
    """
    # Set the model to evaluation mode
    model.eval()

    # Tokenize the text condition
    tokenized = tokenizer(text_condition, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenized.input_ids.to(device)

    # Add prompts using the model's condition_provider
    prompts = model.condition_provider(tokens)

    # Generate music
    with torch.no_grad():
        generated_audio = model.generate(prompts)

    # Save the generated audio as a WAV file
    sf.write(output_path, generated_audio.cpu().numpy(), samplerate=32000)
    print(f"Generated music saved to: {output_path}")
