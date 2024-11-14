#from peft import PromptTuningConfig
#help(PromptTuningConfig)  # 클래스의 사용 가능한 매개변수 확인

import torch

def debug_tensor(tensor, name="Tensor"):
    """
    텐서의 차원, dtype, 장치를 출력하는 디버깅 함수.
    """
    if tensor is None:
        print(f"{name}: None")
        return
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    else:
        print(f"{name}: {tensor} (Not a Tensor)")

def debug_model(model):
    """
    모델의 주요 속성을 출력하는 디버깅 함수.
    """
    print("========== MODEL DEBUG INFO ==========")
    if hasattr(model.lm, "num_codebooks"):
        print(f"model.lm.num_codebooks: {model.lm.num_codebooks}")
    if hasattr(model.lm, "embedding_dim"):
        print(f"model.lm.embedding_dim: {model.lm.embedding_dim}")
    if hasattr(model.condition_provider, "num_virtual_tokens"):
        print(f"model.condition_provider.num_virtual_tokens: {model.condition_provider.num_virtual_tokens}")
    print("======================================")

def debug_training(audio, prompts, outputs):
    """
    학습 과정에서 텐서 정보를 디버깅하기 위한 함수.
    """
    print("========== TRAINING DEBUG INFO ==========")
    debug_tensor(audio, "Audio Tensor")
    debug_tensor(prompts, "Prompts Tensor")
    debug_tensor(outputs, "Outputs Tensor")
    print("=========================================")
