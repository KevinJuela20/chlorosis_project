# Bitemporal Adapter Network (BAN) - Detección de Clorosis Foliar en Supercomputador

Este repositorio contiene la implementación completa en **PyTorch** del enfoque basado en **Bitemporal Adapter Network (BAN)** para la detección y cuantificación de la dinámica temporal de la clorosis foliar, siguiendo la directriz metodológica del director de tesis: **restringir el cálculo del avance de la enfermedad exclusivamente a la zona geométrica de solapamiento de ambas hojas ($\mathcal{M}_{\text{solapamiento}}$)**.

---

## 1. Fundamento Metodológico y Solución al Reto

### El Reto de Investigación
Al comparar hojas en tiempos distintos ($T_1$ inicial vs $T_2$ posterior), existen diferencias de pose, perspectiva y ligera deformación morfológica que generan falsos positivos en los bordes si se realiza una resta directa.

### La Estrategia del Director
1. **Normalización Canónica**: Se utiliza Análisis de Componentes Principales (PCA) sobre el contorno foliar para alinear el eje mayor y centrar ambas hojas a una resolución estándar de **$512 \times 512$**.
2. **Máscara de Solapación ($\mathcal{M}_{\text{solapamiento}}$)**:
   $$\mathcal{M}_{\text{solapamiento}} = \text{Hoja}_{T_1} \cap \text{Hoja}_{T_2}$$
3. **Pérdida Enmascarada (Masked Loss)**:
   El gradiente de retropropagación y la optimización de la red BAN solo se calculan sobre los píxeles dentro de $\mathcal{M}_{\text{solapamiento}}$, ignorando el fondo y las asimetrías de borde:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}^{\text{enmascarada}}(\hat{Y}, Y, \mathcal{M}) + \mathcal{L}_{\text{Dice}}^{\text{enmascarada}}(\hat{Y}, Y, \mathcal{M})$$

---

## 2. Arquitectura de la Red: BANet

La arquitectura implementada en `models/ban_model.py` consta de:
- **Shared Siamese Encoder**: Extrae mapas de características multiescala ($1/2, 1/4, 1/8, 1/16$) compartiendo pesos para $T_1$ y $T_2$.
- **Bitemporal Cross-Attention Adapters**: Módulos de atención cruzada que permiten a las representaciones de $T_1$ consultar el estado morfológico y espectral de $T_2$ y viceversa, capturando gradientes biológicos de decoloración.
- **Difference Enhancement Fusion**: Combina $[F_{T_1}, F_{T_2}, |F_{T_2} - F_{T_1}|]$ para amplificar los cambios patológicos sutiles.
- **FPN Decoder**: Reconstruye la probabilidad de avance de clorosis a resolución nativa $512 \times 512$.

---

## 3. Estructura del Directorio

```text
enfoque_BAN/
├── data/                         # Dataset estructurado a 512x512
│   ├── train/                    # 96 cuartetos bitemporales
│   │   ├── t1/                   # Imágenes T1 (incipiente / sana)
│   │   ├── t2/                   # Imágenes T2 (avanzada)
│   │   ├── overlap/              # Máscaras de solapamiento del director
│   │   └── change/               # Ground truth de avance real
│   ├── val/                      # 16 cuartetos de validación
│   └── test/                     # 24 cuartetos de prueba
├── models/
│   ├── __init__.py
│   ├── ban_model.py              # Arquitectura PyTorch de BANet
│   └── losses.py                 # MaskedDiceLoss y MaskedFocalLoss
├── dataset.py                    # PyTorch Dataset y DataLoaders
├── dataset_prep.py               # Generador y aumentador de pares bitemporales
├── train.py                      # Script de entrenamiento HPC con AMP
├── evaluate.py                   # Script de evaluación y generación de figuras
├── run_hpc.sh                    # Script SLURM para enviar al Supercomputador
├── requirements.txt              # Dependencias de Python
└── README.md                     # Documentación técnica
```

---

## 4. Guía de Ejecución en el Supercomputador (HPC)

### Paso 1: Subir el paquete al Supercomputador
Desde su terminal local:
```bash
scp -r "c:\Users\Fernando\Documents\Proyectos Investigación\Proyecto Clorosis\enfoque_BAN" usuario@hpc.cluster.edu:/home/usuario/
```

### Paso 2: Conectarse y configurar el entorno
```bash
ssh usuario@hpc.cluster.edu
cd enfoque_BAN

# Crear entorno conda (opcional si ya cuenta con PyTorch)
conda create -n env_chlorosis python=3.10 -y
conda activate env_chlorosis
pip install -r requirements.txt
```

### Paso 3: Enviar el trabajo con SLURM
El script `run_hpc.sh` ya tiene configurados los recursos (1 GPU, 8 CPUs, 32 GB RAM, soporte AMP FP16):
```bash
sbatch run_hpc.sh
```

### Paso 4: Monitorear el trabajo
```bash
# Ver estado en cola
squeue -u $USER

# Ver progreso en tiempo real
tail -f logs/ban_*.log
```

### Paso 5: Descargar los resultados
Una vez terminado el trabajo, descargue a su máquina local:
- El modelo entrenado: `checkpoints/best_ban_model.pth`
- El diagnóstico visual: `eval_outputs/resultado_ban_overlap.png`
- El reporte cuantitativo: `eval_outputs/eval_results.json`

```bash
scp usuario@hpc.cluster.edu:/home/usuario/enfoque_BAN/eval_outputs/resultado_ban_overlap.png .
```

---

## 5. Salidas y Diagnóstico Producido

El script `evaluate.py` genera un panel comparativo de 6 columnas por cada par foliar evaluado:
1. **$T_1$**: Estado inicial (clorosis incipiente ~2%).
2. **$T_2$**: Estado posterior (clorosis avanzada ~60-70%).
3. **Solapación del Director**: Región de intersección $\mathcal{M}_{\text{solapamiento}}$ donde se evalúa el experimento.
4. **GT Avance Clorosis**: Avance real de tejido sano a clorótico dentro del solapamiento.
5. **BAN Predicción**: Probabilidad inferida por la red neuronal adaptativa.
6. **Diagnóstico de Error**:
   - 🟩 **Verde**: Verdaderos Positivos (tejido enfermo detectado con precisión).
   - 🟥 **Rojo**: Falsas Alarmas (sobre-estimación).
   - 🟦 **Azul**: Clorosis no detectada (sub-estimación).

