#!/usr/bin/env python3
"""
Training script for ALIGNN with Gated Cross-Attention

Two training strategies:
1. Fine-tune: Freeze graph encoder, only train gated attention (FAST, 5-10 epochs)
2. Full training: Train entire model from scratch (SLOW but BEST, 100 epochs)
"""

import os
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from models.alignn_with_gated_attention import create_gated_alignn


class MaterialsDataset(Dataset):
    """Dataset for materials property prediction."""

    def __init__(self, data_path: str):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate function for batching."""
    import dgl

    graphs = []
    line_graphs = []
    text_list = []
    targets = []

    for sample in batch:
        g = sample.get('graph')
        lg = sample.get('line_graph')

        if isinstance(g, tuple):
            g = g[0]
        if isinstance(lg, tuple):
            lg = lg[0]

        graphs.append(g)
        line_graphs.append(lg)
        text_list.append(sample.get('text', ''))

        target = sample.get('target', 0.0)
        if isinstance(target, (list, np.ndarray)):
            target = target[0]
        targets.append(float(target))

    batched_graph = dgl.batch(graphs)
    batched_line_graph = dgl.batch(line_graphs)
    targets_tensor = torch.FloatTensor(targets)

    return batched_graph, batched_line_graph, text_list, targets_tensor


def encode_text(text_list: List[str], text_encoder, tokenizer, device):
    """Encode text using MatSciBERT."""
    encoded = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = text_encoder(input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state

    text_mask = (attention_mask == 0)

    return text_features, text_mask


def train_epoch(
    model,
    text_encoder,
    tokenizer,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    log_interval=10
):
    """Train for one epoch."""
    model.train()
    text_encoder.eval()  # Keep text encoder frozen

    total_loss = 0
    all_predictions = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch_data in enumerate(pbar):
        g, lg, text_list, targets = batch_data

        g = g.to(device)
        lg = lg.to(device)
        targets = targets.to(device)

        # Encode text
        text_features, text_mask = encode_text(text_list, text_encoder, tokenizer, device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model([g, lg, text_features, text_mask])

        # Compute loss
        loss = criterion(predictions.squeeze(), targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        all_predictions.extend(predictions.detach().cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

        # Update progress bar
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse
    }


def evaluate(
    model,
    text_encoder,
    tokenizer,
    val_loader,
    criterion,
    device
):
    """Evaluate the model."""
    model.eval()
    text_encoder.eval()

    total_loss = 0
    all_predictions = []
    all_targets = []
    all_quality_scores = []

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Evaluating"):
            g, lg, text_list, targets = batch_data

            g = g.to(device)
            lg = lg.to(device)
            targets = targets.to(device)

            # Encode text
            text_features, text_mask = encode_text(text_list, text_encoder, tokenizer, device)

            # Forward pass
            predictions, diagnostics = model([g, lg, text_features, text_mask], return_attention=True)

            # Compute loss
            loss = criterion(predictions.squeeze(), targets)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_quality_scores.extend(diagnostics['text_quality'].cpu().numpy().flatten())

    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    avg_quality = np.mean(all_quality_scores)

    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'avg_text_quality': avg_quality
    }


def train_gated_attention(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    baseline_checkpoint: str,
    text_encoder_path: str = 'm3rg-iitd/matscibert',
    training_mode: str = 'finetune',  # 'finetune' or 'full'
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
):
    """
    Train ALIGNN with Gated Cross-Attention.

    Args:
        training_mode:
            - 'finetune': Freeze graph encoder, only train gated attention (FAST)
            - 'full': Train entire model (SLOW but BEST)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training mode: {training_mode}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load text encoder
    print("Loading text encoder...")
    from transformers import AutoModel, AutoTokenizer

    text_encoder = AutoModel.from_pretrained(text_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False

    # Create model
    print("Creating model with gated attention...")
    model = create_gated_alignn(
        checkpoint_path=baseline_checkpoint,
        hidden_dim=256,
        text_hidden_dim=768,
        use_gated_attention=True,
        gated_attention_layers=1,
        attention_heads=8,
        output_dim=1
    )
    model = model.to(device)

    # Configure training mode
    if training_mode == 'finetune':
        print("Fine-tune mode: Freezing graph encoder, only training gated attention")

        # Freeze graph encoder
        for name, param in model.named_parameters():
            if 'graph_encoder' in name or 'graph_proj' in name:
                param.requires_grad = False

        # Only train gated attention and output head
        trainable_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and ('cross_modal_attention' in n or 'output_head' in n)
        ]

        epochs = min(epochs, 10)  # Fine-tuning doesn't need many epochs
        learning_rate = 5e-4  # Higher LR for fine-tuning

    else:  # full training
        print("Full training mode: Training entire model")
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Create dataloaders
    print("Loading datasets...")
    train_dataset = MaterialsDataset(train_data_path)
    val_dataset = MaterialsDataset(val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Setup optimizer and criterion
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_val_mae = float('inf')
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_r2': [],
        'val_quality': []
    }

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{epochs}")
        print('='*80)

        # Train
        train_metrics = train_epoch(
            model, text_encoder, tokenizer, train_loader,
            optimizer, criterion, device, epoch
        )

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")

        # Validate
        val_metrics = evaluate(
            model, text_encoder, tokenizer, val_loader, criterion, device
        )

        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
              f"R²: {val_metrics['r2']:.4f}")
        print(f"Val   - Avg Text Quality: {val_metrics['avg_text_quality']:.4f}")

        # Update learning rate
        scheduler.step(val_metrics['mae'])

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['val_quality'].append(val_metrics['avg_text_quality'])

        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'val_metrics': val_metrics,
                'training_mode': training_mode
            }

            checkpoint_path = os.path.join(output_dir, 'best_gated_model.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (MAE: {best_val_mae:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)

    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(f"Model saved to: {output_dir}/best_gated_model.pt")
    print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Train ALIGNN with Gated Cross-Attention'
    )

    parser.add_argument(
        '--train_data',
        type=str,
        required=True,
        help='Path to training data (pickle file)'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        required=True,
        help='Path to validation data (pickle file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./gated_training_output',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--baseline_checkpoint',
        type=str,
        required=True,
        help='Path to baseline model checkpoint (for initialization)'
    )
    parser.add_argument(
        '--text_encoder',
        type=str,
        default='m3rg-iitd/matscibert',
        help='MatSciBERT model path'
    )
    parser.add_argument(
        '--training_mode',
        type=str,
        default='finetune',
        choices=['finetune', 'full'],
        help='Training mode: finetune (fast) or full (slow but best)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs (auto-adjusted for finetune mode)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate (auto-adjusted for finetune mode)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()

    train_gated_attention(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        baseline_checkpoint=args.baseline_checkpoint,
        text_encoder_path=args.text_encoder,
        training_mode=args.training_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )


if __name__ == "__main__":
    main()
