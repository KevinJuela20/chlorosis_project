"""
DATASET Y DATALOADER PYTORCH PARA BAN (BITEMPORAL ADAPTER NETWORK)
Proyecto: Cuantificación de Dinámica de Clorosis Foliar en Supercomputador

Carga tuplas bitemporales de 4 elementos:
1. t1: Imagen foliar en tiempo T1 (Tensor 3 x 512 x 512, normalizado)
2. t2: Imagen foliar en tiempo T2 (Tensor 3 x 512 x 512, normalizado)
3. overlap: Máscara geométrica de solapación del director (Tensor 1 x 512 x 512, binario {0, 1})
4. change: Ground-truth de avance de clorosis dentro de overlap (Tensor 1 x 512 x 512, binario {0, 1})
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ChlorosisBitemporalDataset(Dataset):
    """
    Dataset bitemporal que lee las carpetas estructuradas generadas por dataset_prep.py:
    {split}/t1/, {split}/t2/, {split}/overlap/, {split}/change/
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Directorio raíz donde está la carpeta 'data' o la carpeta del split directamente.
            split: 'train', 'val' o 'test'.
            transform: Transformaciones opcionales adicionales.
        """
        self.split_dir = os.path.join(root_dir, split) if os.path.basename(root_dir) != split else root_dir
        self.t1_dir = os.path.join(self.split_dir, 't1')
        self.t2_dir = os.path.join(self.split_dir, 't2')
        self.overlap_dir = os.path.join(self.split_dir, 'overlap')
        self.change_dir = os.path.join(self.split_dir, 'change')

        if not os.path.exists(self.t1_dir):
            raise FileNotFoundError(f"No se encontró el directorio de datos: {self.t1_dir}")

        # Listar todos los IDs de muestra (archivos .png)
        self.sample_ids = sorted([
            f for f in os.listdir(self.t1_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Estadísticas estándar de normalización ImageNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_name = self.sample_ids[idx]
        sample_base = os.path.splitext(sample_name)[0]

        # 1. Cargar Imágenes T1 y T2 (RGB)
        p_t1 = os.path.join(self.t1_dir, sample_name)
        p_t2 = os.path.join(self.t2_dir, sample_name)
        img_t1 = cv2.imread(p_t1)
        img_t2 = cv2.imread(p_t2)

        if img_t1 is None or img_t2 is None:
            raise ValueError(f"Error cargando imágenes bitemporales para la muestra {sample_name}")

        img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Normalización Z-score
        img_t1 = (img_t1 - self.mean) / self.std
        img_t2 = (img_t2 - self.mean) / self.std

        # Formato canal primero: (H, W, C) -> (C, H, W)
        tensor_t1 = torch.from_numpy(img_t1).permute(2, 0, 1).float()
        tensor_t2 = torch.from_numpy(img_t2).permute(2, 0, 1).float()

        # 2. Cargar Máscaras (Grayscale binario {0, 1})
        # Compatibilidad con nombres con o sin prefijo
        p_overlap = os.path.join(self.overlap_dir, f"overlap_{sample_name}")
        if not os.path.exists(p_overlap):
            p_overlap = os.path.join(self.overlap_dir, sample_name)

        p_change = os.path.join(self.change_dir, f"change_{sample_name}")
        if not os.path.exists(p_change):
            p_change = os.path.join(self.change_dir, sample_name)

        mask_overlap = cv2.imread(p_overlap, cv2.IMREAD_GRAYSCALE)
        mask_change = cv2.imread(p_change, cv2.IMREAD_GRAYSCALE)

        if mask_overlap is None:
            mask_overlap = np.ones((img_t1.shape[0], img_t1.shape[1]), dtype=np.uint8) * 255
        if mask_change is None:
            mask_change = np.zeros((img_t1.shape[0], img_t1.shape[1]), dtype=np.uint8)

        # Binarizar estrictamente [0.0, 1.0]
        mask_overlap = (mask_overlap > 127).astype(np.float32)
        mask_change = (mask_change > 127).astype(np.float32)

        tensor_overlap = torch.from_numpy(mask_overlap).unsqueeze(0).float()
        tensor_change = torch.from_numpy(mask_change).unsqueeze(0).float()

        return {
            't1': tensor_t1,
            't2': tensor_t2,
            'overlap': tensor_overlap,
            'change': tensor_change,
            'id': sample_base
        }


def get_dataloaders(data_dir, batch_size=8, num_workers=2, pin_memory=True):
    """
    Construye y retorna los DataLoaders para Train, Val y Test.
    """
    train_dataset = ChlorosisBitemporalDataset(data_dir, split='train')
    val_dataset   = ChlorosisBitemporalDataset(data_dir, split='val')
    test_dataset  = ChlorosisBitemporalDataset(data_dir, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_dataset) > batch_size else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

