from transformers import (
    MusicgenConfig,
    MusicgenDecoderConfig,
    T5Config,
    EncodecConfig,
    MusicgenForConditionalGeneration,
)

# Initializing text encoder, audio encoder, and decoder model configurations
text_encoder_config = T5Config()
audio_encoder_config = EncodecConfig()
decoder_config = MusicgenDecoderConfig()

configuration = MusicgenConfig.from_sub_models_config(
    text_encoder_config, audio_encoder_config, decoder_config
)

# Initializing a MusicgenForConditionalGeneration (with random weights) from the facebook/musicgen-small style configuration
model = MusicgenForConditionalGeneration(configuration)

# Accessing the model configuration
configuration = model.config
config_text_encoder = model.config.text_encoder
config_audio_encoder = model.config.audio_encoder
config_decoder = model.config.decoder

# Saving the model, including its configuration
model.save_pretrained("musicgen-model")

# loading model and config from pretrained folder
musicgen_config = MusicgenConfig.from_pretrained("musicgen-model")
model = MusicgenForConditionalGeneration.from_pretrained("musicgen-model", config=musicgen_config)