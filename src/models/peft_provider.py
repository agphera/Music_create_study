import torch
import torch.nn as nn
from peft import PromptEncoder, PromptEncoderConfig

class PEFTPConditionProvider(nn.Module):
    def __init__(self, prompt_length, hidden_size, num_transformer_submodules, num_attention_heads, num_layers):
        super().__init__()

        self.config = PromptEncoderConfig(
            task_type="TEXT_GENERATION",
            num_virtual_tokens=prompt_length,
            token_dim=hidden_size,
            encoder_hidden_size=hidden_size,
            encoder_num_layers=2, 
            encoder_dropout=0.1,  
            num_transformer_submodules=num_transformer_submodules  # 추가 설정
        )

        self.prompt_encoder = PromptEncoder(self.config)
        self.num_virtual_tokens = prompt_length

    def forward(self, tokens):

        batch_size = tokens.size(0)
        indices = torch.arange(self.num_virtual_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)

        prompt_embeds = self.prompt_encoder(indices)

        if len(prompt_embeds.shape) == 4:  # (1, batch_size, num_virtual_tokens, token_dim)
            prompt_embeds = prompt_embeds.squeeze(0)  # (batch_size, num_virtual_tokens, token_dim)

        # Embed tokens into the same dimension as prompt_embeds
        token_embedding = nn.Embedding(50257, prompt_embeds.size(-1)).to(tokens.device)
        token_embeds = token_embedding(tokens)  # (batch_size, seq_len, token_dim)

        # Concatenate prompts with token embeddings
        return torch.cat([prompt_embeds, token_embeds], dim=1)

