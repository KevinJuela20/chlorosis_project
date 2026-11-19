"""
ARQUITECTURA: BITEMPORAL ADAPTER NETWORK (BAN)
Proyecto: Detección y Cuantificación de Clorosis Foliar en Supercomputador

Componentes:
1. Backbone Siamés con Extracción Jerárquica Multinivel (4 escalas).
2. Bitemporal Cross-Attention Adapters: Permiten intercambio contextual entre T1 y T2.
3. Difference Enhancement Fusion: Fusión de diferencias temporales multi-escala.
4. FPN Decoder con Refinamiento Residual: Reconstrucción a resolución completa (512x512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Bloque convolucional con BatchNorm y activación SiLU / Mish."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class BitemporalCrossAttentionAdapter(nn.Module):
    """
    Adapter Module: Permite que las características de T1 consulten a T2 y viceversa
    mediante Cross-Attention espacial eficiente con conexión residual.
    """
    def __init__(self, channels, reduction=4):
        super(BitemporalCrossAttentionAdapter, self).__init__()
        inter_channels = max(channels // reduction, 16)
        
        # Proyecciones lineales para Query, Key, Value
        self.query_conv = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Factor de escala aprendible para el adaptador
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x_source, x_target):
        """
        x_source 'consulta' el estado de x_target.
        """
        B, C, H, W = x_source.size()
        inter_c = self.query_conv.out_channels
        
        # Reducción espacial si la resolución es grande para acelerar en GPU
        if H > 32:
            pool = nn.AdaptiveAvgPool2d((32, 32))
            x_target_p = pool(x_target)
            _, _, Hp, Wp = x_target_p.size()
            k = self.key_conv(x_target_p).view(B, inter_c, Hp * Wp)
            v = self.value_conv(x_target_p).view(B, C, Hp * Wp)
        else:
            k = self.key_conv(x_target).view(B, inter_c, H * W)
            v = self.value_conv(x_target).view(B, C, H * W)
            
        q = self.query_conv(x_source).view(B, inter_c, H * W).permute(0, 2, 1)
        
        # Matriz de atención cruzada
        attn = torch.bmm(q, k) * (inter_c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Agregación de valores
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        out = self.norm(x_source + self.gamma * out)
        return self.act(out)


class BANet(nn.Module):
    """
    Bitemporal Adapter Network (BAN) completa para supercomputador.
    """
    def __init__(self, in_channels=3, base_channels=32, num_classes=1):
        super(BANet, self).__init__()
        self.num_classes = num_classes
        
        # --- BRAZO SIAMÉS (Shared Encoder) ---
        # Nivel 1: Resolución 1/2 (256x256)
        self.enc1 = ConvBlock(in_channels, base_channels, stride=2)
        # Nivel 2: Resolución 1/4 (128x128)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        # Nivel 3: Resolución 1/8 (64x64)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        # Nivel 4: Resolución 1/16 (32x32)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        
        # --- BITEMPORAL ADAPTERS ---
        self.adapter1 = BitemporalCrossAttentionAdapter(base_channels)
        self.adapter2 = BitemporalCrossAttentionAdapter(base_channels * 2)
        self.adapter3 = BitemporalCrossAttentionAdapter(base_channels * 4)
        self.adapter4 = BitemporalCrossAttentionAdapter(base_channels * 8)
        
        # --- DECODER FPN (Fusión de Diferencias Temporales) ---
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        
        # Canales de diferencia concatenada: [T1, T2, |T2-T1|] -> 3 * C
        self.fuse4 = nn.Sequential(
            nn.Conv2d(c4 * 3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(c3 * 3 + c4, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(c2 * 3 + c3, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c1 * 3 + c2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )
        
        # Capa final de restauración a resolución completa original (512x512)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1 // 2, num_classes, kernel_size=1)
        )

    def forward_encoder(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        return f1, f2, f3, f4

    def forward(self, t1, t2):
        """
        Args:
            t1: Tensor imagen tiempo 1 (B, 3, 512, 512)
            t2: Tensor imagen tiempo 2 (B, 3, 512, 512)
        Returns:
            pred_prob: Tensor de probabilidad de avance (B, 1, 512, 512)
        """
        # 1. Extracción siamesa
        f1_t1, f2_t1, f3_t1, f4_t1 = self.forward_encoder(t1)
        f1_t2, f2_t2, f3_t2, f4_t2 = self.forward_encoder(t2)
        
        # 2. Intercambio contextual bitemporal con Adapters
        a1_t1 = self.adapter1(f1_t1, f1_t2)
        a1_t2 = self.adapter1(f1_t2, f1_t1)
        
        a2_t1 = self.adapter2(f2_t1, f2_t2)
        a2_t2 = self.adapter2(f2_t2, f2_t1)
        
        a3_t1 = self.adapter3(f3_t1, f3_t2)
        a3_t2 = self.adapter3(f3_t2, f3_t1)
        
        a4_t1 = self.adapter4(f4_t1, f4_t2)
        a4_t2 = self.adapter4(f4_t2, f4_t1)
        
        # 3. Concatenación con Diferencia Absoluta Temporal
        diff4 = torch.abs(a4_t2 - a4_t1)
        cat4 = torch.cat([a4_t1, a4_t2, diff4], dim=1)
        d4 = self.fuse4(cat4)
        
        diff3 = torch.abs(a3_t2 - a3_t1)
        d4_up = F.interpolate(d4, size=a3_t1.shape[2:], mode='bilinear', align_corners=True)
        cat3 = torch.cat([a3_t1, a3_t2, diff3, d4_up], dim=1)
        d3 = self.fuse3(cat3)
        
        diff2 = torch.abs(a2_t2 - a2_t1)
        d3_up = F.interpolate(d3, size=a2_t1.shape[2:], mode='bilinear', align_corners=True)
        cat2 = torch.cat([a2_t1, a2_t2, diff2, d3_up], dim=1)
        d2 = self.fuse2(cat2)
        
        diff1 = torch.abs(a1_t2 - a1_t1)
        d2_up = F.interpolate(d2, size=a1_t1.shape[2:], mode='bilinear', align_corners=True)
        cat1 = torch.cat([a1_t1, a1_t2, diff1, d2_up], dim=1)
        d1 = self.fuse1(cat1)
        
        # 4. Proyección a resolución original 512x512
        logits = self.final_upsample(d1)
        pred_prob = torch.sigmoid(logits)
        return pred_prob

