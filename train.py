import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm

from dataset_cassi import CASSIDataset
from ELWRYM_net import ELWRYMNet
from metricas import calcular_psnr, calcular_sam

def calcular_ssim(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0) 
    target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
    ssim_val = ssim_metric(target_np, pred_np, data_range=1.0, channel_axis=-1)
    return float(ssim_val) 

def train_auto():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Automatización en: {device}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta_train = os.path.join(BASE_DIR, "dataset", "fortrain")
    ruta_test = os.path.join(BASE_DIR, "dataset", "fortest")
    
    # Carpeta donde se guardará todo el experimento
    exp_dir = os.path.join(BASE_DIR, "ELWRYM_1a20")
    os.makedirs(exp_dir, exist_ok=True)

    BATCH_SIZE = 8
    EPOCHS_PER_MODEL = 100
    LEARNING_RATE = 1e-4

    print("Cargando datasets...")
    dataset_train = CASSIDataset(root_dir=ruta_train, patch_size=48, num_patches_per_img=50)
    dataset_test = CASSIDataset(root_dir=ruta_test, patch_size=48, num_patches_per_img=1) 
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    
    idx_azar = random.randint(0, len(dataset_test.image_folders) - 1)

    # ==========================================================
    # EL BUCLE MAESTRO: De 1 a 20 bloques
    # ==========================================================
    for num_blocks in range(1, 21):
        print(f"\n{'='*50}")
        print(f"🚀 INICIANDO ENTRENAMIENTO: ELWRYM con {num_blocks} Bloques")
        print(f"{'='*50}")
        
        # 1. Instanciamos un modelo NUEVO cada vez
        modelo = ELWRYMNet(in_channels=1, out_channels=31, num_features=64, num_blocks=num_blocks, S=2).to(device)
        criterio = nn.L1Loss()
        optimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)
        
        hist_loss, hist_psnr, hist_ssim, hist_sam = [], [], [], []

        # 2. Bucle de Épocas (Tqdm por modelo para no ensuciar la terminal)
        loop_epocas = tqdm(range(EPOCHS_PER_MODEL), desc=f"Entrenando {num_blocks} bloques")
        
        for epoch in loop_epocas:
            modelo.train()
            loss_epoch = 0.0
            
            for entrada_2d, objetivo_3d in loader_train:
                entrada_2d, objetivo_3d = entrada_2d.to(device), objetivo_3d.to(device)
                
                optimizador.zero_grad()
                prediccion_3d = modelo(entrada_2d)
                loss = criterio(prediccion_3d, objetivo_3d)
                loss.backward()
                optimizador.step()
                
                loss_epoch += loss.item()

            avg_loss = loss_epoch / len(loader_train)
            hist_loss.append(avg_loss)

            # Validación al final de cada época
            modelo.eval()
            with torch.no_grad():
                test_in_2d, test_gt_3d = dataset_test.get_full_image(idx_azar)
                test_in_2d, test_gt_3d = test_in_2d.to(device), test_gt_3d.to(device)
                
                test_pred_3d = modelo(test_in_2d)
                
                val_psnr = float(calcular_psnr(test_pred_3d, test_gt_3d))
                val_sam = float(calcular_sam(test_pred_3d, test_gt_3d))
                val_ssim = float(calcular_ssim(test_pred_3d[0], test_gt_3d[0]))
                
                hist_psnr.append(val_psnr)
                hist_sam.append(val_sam)
                hist_ssim.append(val_ssim)
            
            # Mostramos las métricas de la última validación en la barra
            loop_epocas.set_postfix(Loss=f"{avg_loss:.4f}", PSNR=f"{val_psnr:.2f}", SSIM=f"{val_ssim:.4f}")

        # 3. Guardamos los pesos y el .npz al finalizar las 100 épocas
        ruta_pesos = os.path.join(exp_dir, f"pesos_elwrym_{num_blocks}_bloques.pth")
        torch.save(modelo.state_dict(), ruta_pesos)
        
        ruta_npz = os.path.join(exp_dir, f"metricas_{num_blocks}_bloques.npz")
        np.savez(ruta_npz, 
                 loss=hist_loss, 
                 psnr=hist_psnr, 
                 ssim=hist_ssim, 
                 sam=hist_sam,
                 bloques=num_blocks)
        
        print(f"✅ Finalizado modelo de {num_blocks} bloques. Datos guardados en {exp_dir}")

if __name__ == "__main__":
    train_auto()