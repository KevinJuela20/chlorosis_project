"""
EVALUACIÓN Y GENERACIÓN DE REPORTES GRÁFICOS: BAN (OVERLAP APPROACH)
Proyecto: Detección y Cuantificación de Clorosis Foliar en Supercomputador

Evalúa el modelo BAN sobre el conjunto de test y produce:
1. Métricas cuantitativas enmascaradas (Masked IoU, Masked Dice, Precision, Recall).
2. Cuantificación de severidad (% de clorosis desarrollada dentro de la zona de solapamiento).
3. Figura científica de alta resolución: 'resultado_ban_overlap.png'.
4. Reporte detallado en JSON: 'eval_results.json'.
"""

import os
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from models.ban_model import BANet
from dataset import ChlorosisBitemporalDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de BAN en Test")
    parser.add_argument('--data-dir', type=str, default='data', help="Directorio con split test")
    parser.add_argument('--model-path', type=str, default='checkpoints/best_ban_model.pth', help="Ruta a los pesos del modelo")
    parser.add_argument('--output-dir', type=str, default='eval_outputs', help="Directorio para guardar gráficos y métricas")
    parser.add_argument('--threshold', type=float, default=0.5, help="Umbral binario para probabilidad BAN")
    parser.add_argument('--device', type=str, default='', help="cuda o cpu")
    return parser.parse_args()


def denormalize_image(tensor):
    """Convierte tensor (3, H, W) normalizado a imagen uint8 (H, W, 3) en [0, 255]."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  EVALUACIÓN CIENTÍFICA BAN - ENFOQUE SOLAPACIÓN DEL DIRECTOR")
    print("=" * 70)
    print(f"Dispositivo         : {device}")
    print(f"Modelo              : {args.model_path}")
    print(f"Directorio de Salida: {args.output_dir}")
    print("=" * 70)

    # 1. Cargar Modelo
    model = BANet(in_channels=3, base_channels=32, num_classes=1)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print(f"Pesos cargados exitosamente desde {args.model_path}")
    else:
        print(f"[AVISO] No se encontró {args.model_path}. Se usará el modelo inicializado para prueba de estructura.")

    model.to(device)
    model.eval()

    # 2. Cargar Dataset de Test
    test_dataset = ChlorosisBitemporalDataset(args.data_dir, split='test')
    print(f"Número de muestras en Test: {len(test_dataset)}")

    # Contenedores de métricas
    ious, dices, precisions, recalls = [], [], [], []
    gt_severities, pred_severities = [], []

    sample_visuals = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            t1 = sample['t1'].unsqueeze(0).to(device)
            t2 = sample['t2'].unsqueeze(0).to(device)
            overlap = sample['overlap'].unsqueeze(0).to(device)
            change = sample['change'].unsqueeze(0).to(device)
            sample_id = sample['id']

            # Inferencia BAN
            pred_prob = model(t1, t2)

            # Restricción ESTRICTA al área de solapamiento del director
            pred_prob_m = pred_prob * overlap
            pred_bin_m = (pred_prob_m > args.threshold).float()
            change_m = change * overlap

            # Métricas por muestra
            tp = (pred_bin_m * change_m).sum().item()
            fp = (pred_bin_m * (1.0 - change_m) * overlap).sum().item()
            fn = ((1.0 - pred_bin_m) * change_m * overlap).sum().item()
            eps = 1e-6

            iou = (tp + eps) / (tp + fp + fn + eps)
            dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
            prec = (tp + eps) / (tp + fp + eps)
            rec = (tp + eps) / (tp + fn + eps)

            ious.append(iou)
            dices.append(dice)
            precisions.append(prec)
            recalls.append(rec)

            # Cuantificación de Severidad dentro del Solapamiento
            overlap_area = overlap.sum().item()
            if overlap_area > 0:
                gt_sev = (change_m.sum().item() / overlap_area) * 100.0
                pred_sev = (pred_bin_m.sum().item() / overlap_area) * 100.0
            else:
                gt_sev = 0.0
                pred_sev = 0.0

            gt_severities.append(gt_sev)
            pred_severities.append(pred_sev)

            # Guardar primeras 3 muestras para visualización
            if len(sample_visuals) < 3:
                sample_visuals.append({
                    'id': sample_id,
                    't1': denormalize_image(t1),
                    't2': denormalize_image(t2),
                    'overlap': (overlap.squeeze().cpu().numpy() * 255).astype(np.uint8),
                    'gt_change': (change_m.squeeze().cpu().numpy() * 255).astype(np.uint8),
                    'pred_change': (pred_bin_m.squeeze().cpu().numpy() * 255).astype(np.uint8),
                    'pred_prob': pred_prob_m.squeeze().cpu().numpy(),
                    'iou': iou,
                    'dice': dice,
                    'gt_sev': gt_sev,
                    'pred_sev': pred_sev
                })

    # Resumen cuantitativo
    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))
    mean_prec = float(np.mean(precisions))
    mean_rec = float(np.mean(recalls))
    mae_severity = float(np.mean(np.abs(np.array(gt_severities) - np.array(pred_severities))))

    results = {
        'test_samples': len(test_dataset),
        'mean_masked_iou': mean_iou,
        'mean_masked_dice': mean_dice,
        'mean_precision': mean_prec,
        'mean_recall': mean_rec,
        'mae_severity_percentage': mae_severity,
        'mean_gt_severity_in_overlap': float(np.mean(gt_severities)),
        'mean_pred_severity_in_overlap': float(np.mean(pred_severities))
    }

    print("\nRESULTADOS EN CONJUNTO DE PRUEBA (TEST):")
    print(f"Masked IoU Promedio        : {mean_iou * 100:.2f}%")
    print(f"Masked Dice (F1) Promedio  : {mean_dice * 100:.2f}%")
    print(f"Precisión en Solapamiento  : {mean_prec * 100:.2f}%")
    print(f"Sensibilidad (Recall)      : {mean_rec * 100:.2f}%")
    print(f"Error Absoluto Medio (MAE) : {mae_severity:.2f}% de severidad foliar")

    # Guardar métricas en JSON
    json_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"\nMétricas guardadas en: {json_path}")

    # 3. Generar Figura Científica de Resultados
    if sample_visuals:
        plt.style.use('default')
        fig, axes = plt.subplots(len(sample_visuals), 6, figsize=(24, 4 * len(sample_visuals)), dpi=200)
        if len(sample_visuals) == 1:
            axes = np.expand_dims(axes, 0)

        for row, sv in enumerate(sample_visuals):
            # 1. Imagen T1
            axes[row, 0].imshow(sv['t1'])
            axes[row, 0].set_title(f"T1: Estado Inicial\n({sv['id']})", fontsize=11, fontweight='bold')
            axes[row, 0].axis('off')

            # 2. Imagen T2
            axes[row, 1].imshow(sv['t2'])
            axes[row, 1].set_title(f"T2: Estado Posterior", fontsize=11, fontweight='bold')
            axes[row, 1].axis('off')

            # 3. Máscara de Solapamiento del Director
            axes[row, 2].imshow(sv['overlap'], cmap='gray')
            axes[row, 2].set_title("Solapación del Director\n(Hoja T1 ∩ Hoja T2)", fontsize=11, fontweight='bold', color='navy')
            axes[row, 2].axis('off')

            # 4. Ground Truth de Avance
            axes[row, 3].imshow(sv['gt_change'], cmap='hot')
            axes[row, 3].set_title(f"GT Avance Clorosis\nSeveridad: {sv['gt_sev']:.1f}%", fontsize=11, fontweight='bold', color='darkred')
            axes[row, 3].axis('off')

            # 5. Predicción BAN
            axes[row, 4].imshow(sv['pred_change'], cmap='hot')
            axes[row, 4].set_title(f"BAN Predicción\nSeveridad: {sv['pred_sev']:.1f}%", fontsize=11, fontweight='bold', color='darkgreen')
            axes[row, 4].axis('off')

            # 6. Mapa de Diagnóstico de Error
            # Verde: TP, Rojo: FP, Azul: FN, Negro: Fondo o Tejido Sano
            h, w = sv['overlap'].shape
            diag = np.zeros((h, w, 3), dtype=np.uint8)
            tp_m = (sv['pred_change'] > 0) & (sv['gt_change'] > 0)
            fp_m = (sv['pred_change'] > 0) & (sv['gt_change'] == 0) & (sv['overlap'] > 0)
            fn_m = (sv['pred_change'] == 0) & (sv['gt_change'] > 0) & (sv['overlap'] > 0)

            diag[tp_m] = [0, 220, 0]    # Verde: Acierto
            diag[fp_m] = [220, 0, 0]    # Rojo: Falsa Alarma
            diag[fn_m] = [0, 100, 255]  # Azul: Clorosis no detectada

            axes[row, 5].imshow(diag)
            axes[row, 5].set_title(f"Diagnóstico en Solapamiento\nIoU: {sv['iou']*100:.1f}% | Dice: {sv['dice']*100:.1f}%", fontsize=11, fontweight='bold')
            axes[row, 5].axis('off')

        plt.suptitle("Bitemporal Adapter Network (BAN) - Detección de Dinámica de Clorosis en Zona de Solapación",
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()

        plot_path = os.path.join(args.output_dir, "resultado_ban_overlap.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Figura de diagnóstico generada: {plot_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

