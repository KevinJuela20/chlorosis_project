"""
PREPARADOR DE DATASET BITEMPORAL CON MÁSCARA DE SOLAPAMIENTO (ROI)
Proyecto: Bitemporal Adapter Network (BAN) en Supercomputador

Estrategia del Director:
- Cada par (T1, T2) se normaliza a resolución 512x512.
- Se genera la Máscara de Solapamiento (Overlap Mask = Hoja_T1 ∩ Hoja_T2).
- El objetivo (Ground Truth) es el tejido sano en T1 que avanzó a clorosis en T2 dentro del solapamiento.
- Se divide en Train, Val y Test con Data Augmentation de rotaciones y flips.
"""

import os
import glob
import cv2
import numpy as np
import shutil

def aislar_hoja_y_clorosis(img_bgr, 
                           lower_blue=(85, 40, 40), 
                           upper_blue=(135, 255, 255),
                           lower_yellow=(22, 148, 108), 
                           upper_yellow=(36, 255, 255)):
    """Segmenta la lámina foliar y la clorosis con los parámetros calibrados."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    mask_fondo = cv2.inRange(hsv, np.array(lower_blue), np.array(upper_blue))
    mask_hoja = cv2.bitwise_not(mask_fondo)
    
    kernel_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_hoja = cv2.morphologyEx(mask_hoja, cv2.MORPH_OPEN, kernel_h)
    mask_hoja = cv2.morphologyEx(mask_hoja, cv2.MORPH_CLOSE, kernel_h)
    
    mask_amarillo = cv2.inRange(hsv, np.array(lower_yellow), np.array(upper_yellow))
    mask_clorosis = cv2.bitwise_and(mask_amarillo, mask_hoja)
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clorosis = cv2.morphologyEx(mask_clorosis, cv2.MORPH_OPEN, kernel_e)
    
    return mask_hoja, mask_clorosis

def normalizar_canonica(img_bgr, mask_hoja, mask_clorosis, target_size=(512, 512)):
    """
    Alinea la hoja por su eje principal (PCA de contorno), la centra y normaliza
    a resolución de alta fidelidad 512x512 para el supercomputador.
    """
    cnts, _ = cv2.findContours(mask_hoja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No se detectó contorno foliar.")
        
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] == 0:
        raise ValueError("Momento cero en la hoja.")
        
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    
    pts = c.reshape(-1, 2).astype(np.float32)
    _, eigenvectors = cv2.PCACompute(pts, mean=np.array([[cx, cy]], dtype=np.float32))
    angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    
    rect = cv2.minAreaRect(c)
    w_box, h_box = rect[1]
    longitud = max(w_box, h_box)
    
    h_target, w_target = target_size
    target_long = h_target * 0.82
    escala = target_long / max(longitud, 1.0)
    
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle - 90, escala)
    rot_mat[0, 2] += (w_target / 2 - cx)
    rot_mat[1, 2] += (h_target / 2 - cy)
    
    c_img = cv2.warpAffine(img_bgr, rot_mat, (w_target, h_target), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    c_hoja = cv2.warpAffine(mask_hoja, rot_mat, (w_target, h_target), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    c_clor = cv2.warpAffine(mask_clorosis, rot_mat, (w_target, h_target), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return c_img, c_hoja, c_clor

def rotar_todo(t1, t2, overlap, change, angle):
    """Rota conjuntamente las 4 matrices del par para mantener correspondencia exacta."""
    h, w = t1.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    
    t1_r = cv2.warpAffine(t1, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    t2_r = cv2.warpAffine(t2, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    ov_r = cv2.warpAffine(overlap, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    ch_r = cv2.warpAffine(change, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return t1_r, t2_r, ov_r, ch_r

def flip_todo(t1, t2, overlap, change, flip_code):
    """Aplica flip horizontal o vertical a las 4 matrices simultáneamente."""
    return (
        cv2.flip(t1, flip_code),
        cv2.flip(t2, flip_code),
        cv2.flip(overlap, flip_code),
        cv2.flip(change, flip_code)
    )

def preparar_pares_ban(source_dir="..", output_dir="data", target_size=(512, 512)):
    print("=" * 70)
    print("PREPARACIÓN DEL DATASET BAN CON MÁSCARA DE SOLAPAMIENTO (512x512)")
    print("=" * 70)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    for split in ["train", "val", "test"]:
        for sub in ["t1", "t2", "overlap", "change"]:
            os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)
            
    # Asignación de pares bitemporales representativos
    # Test contiene explícitamente el par clave: Img 1 (incipiente) -> Img 2 (66% avanzada)
    pares_split = {
        "test": [
            ("Img 1.jpeg", "Img 2.jpeg"),   # Par objetivo de investigación
            ("Img 1.jpeg", "Img 15.jpeg"),
            ("Img 3.jpeg", "Img 14.jpeg")
        ],
        "val": [
            ("Img 6.jpeg", "Img 13.jpeg"),
            ("Img 12.jpeg", "Img 14.jpeg")
        ],
        "train": [
            ("Img 3.jpeg", "Img 4.jpeg"),
            ("Img 3.jpeg", "Img 7.jpeg"),
            ("Img 3.jpeg", "Img 8.jpeg"),
            ("Img 5.jpeg", "Img 9.jpeg"),
            ("Img 5.jpeg", "Img 10.jpeg"),
            ("Img 6.jpeg", "Img 7.jpeg"),
            ("Img 6.jpeg", "Img 11.jpeg"),
            ("Img 12.jpeg", "Img 4.jpeg"),
            ("Img 12.jpeg", "Img 8.jpeg"),
            ("Img 12.jpeg", "Img 9.jpeg"),
            ("Img 5.jpeg", "Img 11.jpeg"),
            ("Img 6.jpeg", "Img 10.jpeg")
        ]
    }
    
    angulos = [0, 90, 180, 270]
    total_muestras = {"train": 0, "val": 0, "test": 0}
    
    for split, pares in pares_split.items():
        print(f"\n--- Procesando Split: {split.upper()} ({len(pares)} pares base) ---")
        for f1_name, f2_name in pares:
            p1 = os.path.join(source_dir, f1_name)
            p2 = os.path.join(source_dir, f2_name)
            
            if not os.path.exists(p1) or not os.path.exists(p2):
                print(f"[Aviso] No se encontró el par: {f1_name} <-> {f2_name}")
                continue
                
            img1 = cv2.imread(p1)
            img2 = cv2.imread(p2)
            
            # Segmentar ambas
            h1, c1 = aislar_hoja_y_clorosis(img1)
            h2, c2 = aislar_hoja_y_clorosis(img2)
            
            # Normalizar canónicamente a 512x512
            can_t1, can_h1, can_c1 = normalizar_canonica(img1, h1, c1, target_size=target_size)
            can_t2, can_h2, can_c2 = normalizar_canonica(img2, h2, c2, target_size=target_size)
            
            # 1. MÁSCARA DE SOLAPAMIENTO DEL DIRECTOR: Intersección de ambas láminas foliares
            overlap = cv2.bitwise_and(can_h1, can_h2)
            area_overlap = cv2.countNonZero(overlap)
            
            if area_overlap == 0:
                continue
                
            # 2. GROUND TRUTH DE AVANCE: Sano en T1 que pasó a Clorosis en T2 dentro del solapamiento
            sano_t1 = cv2.bitwise_and(cv2.bitwise_not(can_c1), can_h1)
            change = cv2.bitwise_and(sano_t1, can_c2)
            change = cv2.bitwise_and(change, overlap)  # Restringido estrictamente al solapamiento
            
            area_change = cv2.countNonZero(change)
            pct_avance = (area_change / area_overlap) * 100.0
            
            base_id = f"{os.path.splitext(f1_name)[0]}_to_{os.path.splitext(f2_name)[0]}"
            print(f"  • {base_id:28s} -> Solapamiento: {area_overlap:,} px | Avance neto: {pct_avance:5.2f}%")
            
            # Data Augmentation: rotaciones + flips
            for ang in angulos:
                t1_r, t2_r, ov_r, ch_r = rotar_todo(can_t1, can_t2, overlap, change, ang)
                
                for flip_mode, suffix in [(None, ""), (1, "_fliph")]:
                    if flip_mode is not None:
                        t1_f, t2_f, ov_f, ch_f = flip_todo(t1_r, t2_r, ov_r, ch_r, flip_mode)
                    else:
                        t1_f, t2_f, ov_f, ch_f = t1_r, t2_r, ov_r, ch_r
                        
                    nombre = f"{base_id}_rot{ang}{suffix}.png"
                    
                    cv2.imwrite(os.path.join(output_dir, split, "t1", nombre), t1_f)
                    cv2.imwrite(os.path.join(output_dir, split, "t2", nombre), t2_f)
                    cv2.imwrite(os.path.join(output_dir, split, "overlap", nombre), ov_f)
                    cv2.imwrite(os.path.join(output_dir, split, "change", nombre), ch_f)
                    total_muestras[split] += 1

    print("\n" + "=" * 70)
    print("RESUMEN DE GENERACIÓN PARA BAN (SUPERCOMPUTADOR)")
    print("=" * 70)
    print(f"Directorio de salida: {os.path.abspath(output_dir)}")
    print(f"  • Train: {total_muestras['train']} cuaternas (T1, T2, Overlap, Change)")
    print(f"  • Val:   {total_muestras['val']} cuaternas")
    print(f"  • Test:  {total_muestras['test']} cuaternas")
    print(f"Total muestras bitemporales en 512x512: {sum(total_muestras.values())}")
    print("=" * 70)

if __name__ == "__main__":
    preparar_pares_ban()

