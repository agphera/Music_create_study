import torch
from torch.optim import AdamW
import os
from debug import debug_tensor, debug_training, debug_model

def train_model(model, tokenizer, dataloader, device, epochs, grad_acc_steps, lr, checkpoint_dir):
    # Optimizer 설정
    optimizer = AdamW(
        list(model.lm.parameters()) + list(model.condition_provider.parameters()), 
        lr=lr
    )
    # 손실 함수 정의
    loss_fn = torch.nn.MSELoss()

    # 모델 디버깅
    debug_model(model)

    model.lm.train()
    model.condition_provider.train()

    for epoch in range(epochs):
        total_loss = 0

        for i, (audio, text) in enumerate(dataloader):
            # GPU/CPU로 이동
            audio = audio.to(device)
            text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            # 텐서 디버깅
            debug_tensor(audio, "Audio Tensor (Original)")
            debug_tensor(text_tokens, "Text Tokens")

            # 프롬프트 생성
            prompts = model.condition_provider(text_tokens)

            # 입력 텐서를 모델이 요구하는 형태로 전처리
            # 모델이 요구하는 num_codebooks와 hidden_size를 가져옴
            num_codebooks = model.lm.num_codebooks
            hidden_size = model.lm.embedding_dim if hasattr(model.lm, "embedding_dim") else 768

            # Audio 텐서를 num_codebooks에 맞게 확장
            audio = audio.unsqueeze(1).expand(-1, num_codebooks, -1)

            # 텐서를 정수형으로 변환
            audio = audio.to(torch.long)

            # 디버깅: 변환 후 audio와 prompts 확인
            debug_tensor(audio, "Audio Tensor (Processed for Model)")
            debug_tensor(prompts, "Prompts Tensor")

            # 모델의 forward pass
            try:
                outputs = model.lm(audio, prompts)
            except Exception as e:
                # 에러 발생 시 디버깅 정보 출력
                print(f"Error during training step: {e}")
                debug_tensor(audio, "Audio Tensor (Error Context)")
                debug_tensor(prompts, "Prompts Tensor (Error Context)")
                raise e

            # 출력 디버깅
            debug_training(audio, prompts, outputs)

            # 손실 계산
            loss = loss_fn(outputs, audio)  # 실제 loss 함수는 모델 출력 형식에 따라 다를 수 있음
            total_loss += loss.item()

            # 역전파
            loss.backward()
            if (i + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

        # 체크포인트 저장
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.lm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)
