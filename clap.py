from transformers import AutoProcessor, ClapModel
import torchaudio
import torch
import torchaudio.transforms as transforms

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CLAP 모델과 프로세서 로드
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
model.to(device)
model.eval()

# 오디오 데이터를 리샘플링하는 함수
def resample_audio(waveform, original_sample_rate, target_sample_rate=48000):
    if original_sample_rate != target_sample_rate:
        resampler = transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform

# 오디오 데이터를 자르는 함수
def truncate_waveform(waveform, sample_rate, max_duration=10):
    max_samples = int(sample_rate * max_duration)  # 최대 샘플 수
    return waveform[:, :max_samples]  # 초과 데이터 제거

# 유사도 계산 함수 정의
def compute_similarity(text, audio_path, max_duration=10):
    # 텍스트 임베딩 생성
    text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    # 오디오 데이터 로드
    waveform, sample_rate = torchaudio.load(audio_path)

    # 리샘플링 및 트리밍
    waveform = resample_audio(waveform, sample_rate, target_sample_rate=48000)
    waveform = truncate_waveform(waveform, sample_rate=48000, max_duration=max_duration)

    # 오디오 임베딩 생성
    audio_inputs = processor(audios=waveform, sampling_rate=48000, return_tensors="pt").to(device)
    with torch.no_grad():
        audio_features = model.get_audio_features(**audio_inputs)

    # 코사인 유사도 계산
    similarity = torch.nn.functional.cosine_similarity(text_features, audio_features)
    return similarity.item()

# 평가할 텍스트와 오디오 파일 경로
text_prompt = "calm and cozy christmas jazz with piano and lofi style"
generated_audio_path = "ptuned_gen.wav"  # 생성된 오디오 파일 경로

# 유사도 계산 및 출력
similarity_score = compute_similarity(text_prompt, generated_audio_path, max_duration=10)
print(f"Similarity between generated audio and text: {similarity_score}")
