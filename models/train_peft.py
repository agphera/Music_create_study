# models/trainer.py

import torch
from torch.optim import AdamW
import wandb
import os

def train_model(model, tokenizer, dataloader, device, epochs, grad_acc_steps, lr, checkpoint_dir):
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    wandb.login()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (audio, text) in enumerate(dataloader):
            tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            tokens = tokenized.input_ids.to(device)

            prompts = model.condition_provider(tokens)
            outputs = model(audio.to(device), prompts)

            loss = loss_fn(outputs.generated_audio, audio.to(device))
            total_loss += loss.item()

            loss.backward()
            if (i + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            wandb.log({"epoch": epoch, "loss": loss.item()})

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(model.state_dict(), f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
