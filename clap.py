import numpy as np
import librosa
import torch
import laion_clap


# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

# 코사인 유사도 계산 함수
def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1 / tensor1.norm(dim=-1, keepdim=True)
    tensor2 = tensor2 / tensor2.norm(dim=-1, keepdim=True)
    return torch.matmul(tensor1, tensor2.T)

# CLAP 모델 로드
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

# 오디오 및 텍스트 데이터
audio_file = [

    #'/Users/dana/Desktop/Music_study/ptuned_mlp1.wav' #0.4515
    #'/Users/dana/Desktop/Music_study/origin_1.wav' #0.3916

    #"A warm lo-fi track with gentle beats and winter jazz style"
    #'/Users/dana/Desktop/Music_study/origin_2.wav' #0.4166
    #'/Users/dana/Desktop/Music_study/ptuned_mlp2_1.wav' #0.4323

]

#text_data=["cozy and dreamy christmas jazz music"]
text_data = ["A warm lo-fi track with gentle beats and winter jazz style"]

# 오디오 임베딩 생성
audio_embed = model.get_audio_embedding_from_filelist(x=audio_file, use_tensor=True)
print(f"Audio Embeddings: {audio_embed.shape}")

# 텍스트 임베딩 생성
text_embed = model.get_text_embedding(text_data, use_tensor=True)
print(f"Text Embeddings: {text_embed.shape}")

# CLAP 점수 계산
clap_scores = cosine_similarity(audio_embed, text_embed)
print("CLAP Scores (Audio vs. Text):")
print(clap_scores)