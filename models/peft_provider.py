# models/peft_provider.py

from peft import PromptEncoderConfig, PromptEncoder
import torch
import torch.nn as nn

class PEFTPConditionProvider(nn.Module):
    def __init__(self, prompt_length, hidden_size, num_transformer_submodules, num_attention_heads, num_layers):
        super().__init__()

        self.config = PromptEncoderConfig(
            task_type="TEXT_GENERATION",  # 작업 유형
            num_virtual_tokens=prompt_length,  # 학습 가능한 가상 토큰의 수
            token_dim=hidden_size,  # 모델의 hidden_size와 동일
            num_transformer_submodules=num_transformer_submodules,  # Transformer 서브모듈 수
            num_attention_heads=num_attention_heads,  # Attention 헤드 수
            num_layers=num_layers,  # 레이어 수
            encoder_hidden_size=hidden_size,  # Prompt 인코더의 hidden_size
            encoder_num_layers=2,  # Prompt 인코더 레이어 수
            encoder_dropout=0.1  # Dropout 확률
        )
        self.prompt_encoder = PromptEncoder(self.config)

    def forward(self, tokens):
        batch_size = tokens.size(0)
        prompt_embeds = self.prompt_encoder()
        prompt_expanded = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_expanded, tokens], dim=1)
