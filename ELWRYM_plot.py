import os
import numpy as np
import matplotlib.pyplot as plt

def plot_experiment_results():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(BASE_DIR, "ELWRYM_1a20")
    
    if not os.path.exists(exp_dir):
        print(f"La carpeta {exp_dir} no existe. ¡Debes correr el entrenamiento primero!")
        return

    num_bloques_list = []
    
    # Listas para guardar el PROMEDIO
    mean_psnr, mean_ssim, mean_sam = [], [], []
    
    # Listas para guardar la DESVIACIÓN ESTÁNDAR (Barras de error)
    std_psnr, std_ssim, std_sam = [], [], []

    # Vamos a promediar las últimas 10 épocas para ver la estabilidad de convergencia
    N_ULTIMAS = 10 

    for i in range(1, 21):
        ruta_npz = os.path.join(exp_dir, f"metricas_{i}_bloques.npz")
        if os.path.exists(ruta_npz):
            data = np.load(ruta_npz)
            num_bloques_list.append(data['bloques'])
            
            # 1. Extraemos solo las últimas 10 épocas
            ultimos_psnr = data['psnr'][-N_ULTIMAS:]
            ultimos_ssim = data['ssim'][-N_ULTIMAS:]
            ultimos_sam = data['sam'][-N_ULTIMAS:]
            
            # 2. Calculamos el promedio (Media)
            mean_psnr.append(np.mean(ultimos_psnr))
            mean_ssim.append(np.mean(ultimos_ssim))
            mean_sam.append(np.mean(ultimos_sam))
            
            # 3. Calculamos la desviación estándar (El tamaño de la barra de error)
            std_psnr.append(np.std(ultimos_psnr))
            std_ssim.append(np.std(ultimos_ssim))
            std_sam.append(np.std(ultimos_sam))
        else:
            print(f"Falta el archivo: {ruta_npz}")

    if not num_bloques_list:
        print("No se encontraron archivos .npz para graficar.")
        return

    # Creamos la figura
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'Estudio de Ablación ELWRYM: Promedio y Estabilidad (Últimas {N_ULTIMAS} Épocas)', 
                 fontsize=15, fontweight='bold')

    # Parámetros visuales para las barras de error
    formato_linea = {'linewidth': 2, 'markersize': 6}
    formato_error = {'ecolor': 'red', 'capsize': 4, 'elinewidth': 1.5, 'markeredgewidth': 1.5}

    # Gráfico 1: PSNR
    ax1.errorbar(num_bloques_list, mean_psnr, yerr=std_psnr, fmt='-o', color='blue', **formato_error, **formato_linea)
    ax1.set_ylabel('PSNR (dB) - Más alto es mejor', fontsize=12)
    ax1.set_xticks(num_bloques_list)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Gráfico 2: SSIM
    ax2.errorbar(num_bloques_list, mean_ssim, yerr=std_ssim, fmt='-s', color='green', **formato_error, **formato_linea)
    ax2.set_ylabel('SSIM - Más alto es mejor', fontsize=12)
    ax2.set_xticks(num_bloques_list)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Gráfico 3: SAM (Corregido a grados)
    ax3.errorbar(num_bloques_list, mean_sam, yerr=std_sam, fmt='-^', color='purple', **formato_error, **formato_linea)
    ax3.set_xlabel('Número de Bloques DFHM', fontsize=12)
    ax3.set_ylabel('SAM (grados) - Más bajo es mejor', fontsize=12)
    ax3.set_xticks(num_bloques_list)
    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Guardamos la nueva imagen
    ruta_plot = os.path.join(exp_dir, "resultados_ablacion_elwrym_con_error.png")
    plt.savefig(ruta_plot, dpi=300)
    print(f"✅ Gráfico con barras de error guardado en: {ruta_plot}")
    plt.show()

if __name__ == "__main__":
    plot_experiment_results()