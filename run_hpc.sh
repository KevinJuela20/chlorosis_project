#!/bin/bash
# ==============================================================================
# SCRIPT DE LANZAMIENTO SLURM PARA SUPERCOMPUTADOR (HPC)
# Proyecto: Bitemporal Adapter Network (BAN) - Dinámica de Clorosis Foliar
# Directorio: enfoque_BAN
# ==============================================================================

#SBATCH --job-name=BAN_Chlorosis
#SBATCH --output=logs/ban_%j.log
#SBATCH --error=logs/ban_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

echo "========================================================================"
echo " INICIO DEL TRABAJO HPC: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo " Nodo Asignado       : $SLURMD_NODENAME"
echo " Fecha y Hora        : $(date)"
echo " Directorio Actual   : $(pwd)"
echo "========================================================================"

# 1. Crear directorios para logs, checkpoints y salidas
mkdir -p logs checkpoints eval_outputs

# 2. Configurar entorno de software / modulos HPC
# (Ajuste los módulos según la distribución específica de su clúster HPC, e.g. Slurm/CentOS/Rocky/Ubuntu)
if command -v module &> /dev/null; then
    echo "[HPC] Cargando módulos del sistema..."
    module purge
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
    module load python/3.10 2>/dev/null || module load python/3.11 2>/dev/null || true
fi

# Activar entorno Conda si existe
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate env_chlorosis 2>/dev/null || conda activate ban_env 2>/dev/null || true
    echo "[HPC] Entorno conda activado: $CONDA_DEFAULT_ENV"
elif [ -d "$HOME/miniconda3/envs/env_chlorosis" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate env_chlorosis
    echo "[HPC] Entorno conda 'env_chlorosis' activado."
elif [ -d "$HOME/anaconda3/envs/env_chlorosis" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate env_chlorosis
    echo "[HPC] Entorno conda 'env_chlorosis' activado."
elif [ -d "$HOME/miniconda3/envs/ban_env" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ban_env
    echo "[HPC] Entorno conda 'ban_env' activado."
fi

# Diagnóstico de GPU
echo "========================================================================"
echo " DIAGNÓSTICO DE GPU ASIGNADA"
echo "========================================================================"
nvidia-smi
echo "========================================================================"

# 3. Instalación rápida de dependencias si es necesario
# pip install -r requirements.txt --quiet

# 4. Fase 1: Entrenamiento del Modelo BAN con Solapación Enmascarada
echo "[HPC] >> INICIANDO ENTRENAMIENTO BAN (512x512, AMP FP16)..."
python train.py \
    --data-dir data \
    --save-dir checkpoints \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --num-workers 8 \
    --amp

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[HPC ERROR] El entrenamiento falló con código de salida: $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# 5. Fase 2: Evaluación Científica y Generación de Gráficos
echo "[HPC] >> INICIANDO EVALUACIÓN CIENTÍFICA SOBRE CONJUNTO DE TEST..."
python evaluate.py \
    --data-dir data \
    --model-path checkpoints/best_ban_model.pth \
    --output-dir eval_outputs \
    --threshold 0.5

EVAL_EXIT=$?
if [ $EVAL_EXIT -ne 0 ]; then
    echo "[HPC ERROR] La evaluación falló con código de salida: $EVAL_EXIT"
    exit $EVAL_EXIT
fi

echo "========================================================================"
echo " TRABAJO HPC COMPLETADO CON ÉXITO"
echo " Pesos óptimos guardados en : checkpoints/best_ban_model.pth"
echo " Métricas y Gráficos en     : eval_outputs/"
echo " Fecha y Hora de Fin        : $(date)"
echo "========================================================================"

