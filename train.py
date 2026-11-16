"""
SCRIPT DE ENTRENAMIENTO HPC: BITEMPORAL ADAPTER NETWORK (BAN)
Proyecto: Detección y Cuantificación de Clorosis Foliar en Supercomputador

Características para Supercomputador:
- Soporte para Mixed Precision (FP16 / AMP) con torch.cuda.amp.GradScaler.
- Optimización con AdamW y scheduler CosineAnnealingLR.
- Pérdida y Métricas Enmascaradas (Masked Loss / Masked IoU / Masked Dice) restringidas
  estrictamente a la región de solapación geométrica determinada por la estrategia del director.
- Checkpointing automático (guarda el mejor modelo por Masked Dice y el último checkpoint).
- Registro de métricas en JSON y CSV para análisis de convergencia.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Importar arquitectura, pérdidas y dataset propios
from models.ban_model import BANet
from models.losses import MaskedCompoundLoss
from dataset import get_dataloaders


def compute_masked_metrics(preds, targets, masks_overlap, threshold=0.5, eps=1e-6):
    """
    Calcula IoU, Dice, Precision y Recall calculados estrictamente dentro del solapamiento.
    """
    preds_bin = (preds > threshold).float() * masks_overlap
    targets_bin = targets * masks_overlap

    # True Positives, False Positives, False Negatives
    tp = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
    fp = (preds_bin * (1.0 - targets_bin) * masks_overlap).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds_bin) * targets_bin * masks_overlap).sum(dim=(1, 2, 3))

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento BAN en Supercomputador")
    parser.add_argument('--data-dir', type=str, default='data', help="Directorio con subcarpetas train, val, test")
    parser.add_argument('--save-dir', type=str, default='checkpoints', help="Directorio para almacenar pesos y logs")
    parser.add_argument('--epochs', type=int, default=50, help="Número de épocas de entrenamiento")
    parser.add_argument('--batch-size', type=int, default=8, help="Tamaño de lote (batch size)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate inicial")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="Decaimiento de pesos AdamW")
    parser.add_argument('--num-workers', type=int, default=4, help="Número de hilos para DataLoader")
    parser.add_argument('--amp', action='store_true', default=True, help="Habilitar Automatic Mixed Precision (AMP FP16)")
    parser.add_argument('--device', type=str, default='', help="cuda o cpu (vacio = autodetectar)")
    parser.add_argument('--seed', type=int, default=42, help="Semilla para reproducibilidad científica")
    return parser.parse_args()


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    running_focal = 0.0
    running_dice = 0.0
    metrics_sum = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
    num_batches = 0

    for batch in loader:
        t1 = batch['t1'].to(device, non_blocking=True)
        t2 = batch['t2'].to(device, non_blocking=True)
        overlap = batch['overlap'].to(device, non_blocking=True)
        change = batch['change'].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward con AMP si está habilitado en CUDA
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                pred = model(t1, t2)
                loss, focal, dice = criterion(pred, change, overlap)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(t1, t2)
            loss, focal, dice = criterion(pred, change, overlap)
            loss.backward()
            optimizer.step()

        # Métricas
        with torch.no_grad():
            m = compute_masked_metrics(pred, change, overlap)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

        running_loss += loss.item()
        running_focal += focal.item()
        running_dice += dice.item()
        num_batches += 1

    return {
        'loss': running_loss / max(num_batches, 1),
        'focal': running_focal / max(num_batches, 1),
        'dice_loss': running_dice / max(num_batches, 1),
        'iou': metrics_sum['iou'] / max(num_batches, 1),
        'dice': metrics_sum['dice'] / max(num_batches, 1),
        'precision': metrics_sum['precision'] / max(num_batches, 1),
        'recall': metrics_sum['recall'] / max(num_batches, 1)
    }


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    running_focal = 0.0
    running_dice = 0.0
    metrics_sum = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            t1 = batch['t1'].to(device, non_blocking=True)
            t2 = batch['t2'].to(device, non_blocking=True)
            overlap = batch['overlap'].to(device, non_blocking=True)
            change = batch['change'].to(device, non_blocking=True)

            if use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred = model(t1, t2)
                    loss, focal, dice = criterion(pred, change, overlap)
            else:
                pred = model(t1, t2)
                loss, focal, dice = criterion(pred, change, overlap)

            m = compute_masked_metrics(pred, change, overlap)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

            running_loss += loss.item()
            running_focal += focal.item()
            running_dice += dice.item()
            num_batches += 1

    return {
        'loss': running_loss / max(num_batches, 1),
        'focal': running_focal / max(num_batches, 1),
        'dice_loss': running_dice / max(num_batches, 1),
        'iou': metrics_sum['iou'] / max(num_batches, 1),
        'dice': metrics_sum['dice'] / max(num_batches, 1),
        'precision': metrics_sum['precision'] / max(num_batches, 1),
        'recall': metrics_sum['recall'] / max(num_batches, 1)
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Configuración de dispositivo
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  ENTRENAMIENTO BITEMPORAL ADAPTER NETWORK (BAN) - ENFOQUE OVERLAP")
    print("=" * 70)
    print(f"Dispositivo activo      : {device}")
    if device.type == 'cuda':
        print(f"Nombre de GPU           : {torch.cuda.get_device_name(0)}")
        print(f"Memoria VRAM Total      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Directorio de Datos     : {args.data_dir}")
    print(f"Batch Size              : {args.batch_size}")
    print(f"Épocas                  : {args.epochs}")
    print(f"Learning Rate Inicial   : {args.lr}")
    print(f"Mixed Precision (AMP)   : {args.amp and device.type == 'cuda'}")
    print("=" * 70)

    # 1. Cargar DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    print(f"Muestras de Entrenamiento: {len(train_loader.dataset)}")
    print(f"Muestras de Validación   : {len(val_loader.dataset)}")
    print(f"Muestras de Prueba       : {len(test_loader.dataset)}")

    # 2. Inicializar Modelo BAN
    model = BANet(in_channels=3, base_channels=32, num_classes=1)
    model.to(device)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros Totales BAN  : {total_params:,} ({trainable_params:,} entrenables)")

    # 3. Pérdida y Optimizador
    criterion = MaskedCompoundLoss(weight_focal=1.0, weight_dice=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None

    # Historial de entrenamiento
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': []
    }

    best_val_dice = -1.0
    best_epoch = 0
    start_time = time.time()

    print("\nIniciando optimización...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.amp)
        val_stats = evaluate(model, val_loader, criterion, device, args.amp)
        scheduler.step()

        elapsed = time.time() - epoch_start

        # Guardar historial
        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_stats['loss'])
        history['train_iou'].append(train_stats['iou'])
        history['val_iou'].append(val_stats['iou'])
        history['train_dice'].append(train_stats['dice'])
        history['val_dice'].append(val_stats['dice'])

        # Log por época
        print(f"Época [{epoch:03d}/{args.epochs:03d}] ({elapsed:.1f}s) | "
              f"Tr Loss: {train_stats['loss']:.4f} (IoU: {train_stats['iou']*100:.1f}%, Dice: {train_stats['dice']*100:.1f}%) | "
              f"Val Loss: {val_stats['loss']:.4f} (IoU: {val_stats['iou']*100:.1f}%, Dice: {val_stats['dice']*100:.1f}%) | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Guardar mejor modelo según Masked Dice en Validación
        if val_stats['dice'] > best_val_dice:
            best_val_dice = val_stats['dice']
            best_epoch = epoch
            best_ckpt_path = os.path.join(args.save_dir, "best_ban_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_stats,
                'args': vars(args)
            }, best_ckpt_path)

        # Guardar último modelo
        latest_ckpt_path = os.path.join(args.save_dir, "latest_ban_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'args': vars(args)
        }, latest_ckpt_path)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Entrenamiento Finalizado en {total_time / 60:.2f} minutos.")
    print(f"Mejor Época: {best_epoch} con Masked Dice: {best_val_dice * 100:.2f}%")
    print(f"Pesos óptimos guardados en: {os.path.join(args.save_dir, 'best_ban_model.pth')}")

    # Guardar historial en JSON
    history_path = os.path.join(args.save_dir, "train_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    print(f"Historial exportado a: {history_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

