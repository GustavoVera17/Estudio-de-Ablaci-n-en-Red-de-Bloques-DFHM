# ELWRYM: Extra Lightweight Red Yamawaki Model - Estudio de Ablación

Este repositorio contiene la implementación y el estudio de profundidad funcional de **ELWRYM** (~0.34 MB), una variante altamente optimizada de la arquitectura de reconstrucción hiperespectral propuesta originalmente por Yamawaki et al. (que pesa ~42 MB).

## 🔬 El Experimento: Profundidad vs. Estabilidad
El objetivo de este proyecto es demostrar empíricamente el impacto de apilar secuencialmente módulos **DFHM** (Deep Feature Hallucination Modules) en un entorno de compresión extrema (1x1) y convoluciones *depth-wise*.

Se ejecutó un entrenamiento automatizado (inicialización en frío) generando 20 modelos independientes, variando la profundidad desde **1 hasta 20 bloques DFHM**. Cada modelo fue entrenado durante 100 épocas bajo condiciones idénticas.

## 📊 Hallazgos Clave
El gráfico de resultados (con barras de error calculadas sobre las últimas 10 épocas de convergencia) revela hallazgos críticos sobre la arquitectura:

1. **El "Sweet Spot" (Bloques 9 y 10):** La arquitectura alcanza su máximo nivel de rendimiento y estabilidad estructural entre los 9 y 10 bloques. Presentan los promedios más altos de PSNR (~24.4 dB) y barras de error notablemente estrechas, indicando una convergencia sólida.
2. **Saturación y Caos:** Aumentar la profundidad más allá de los 11 bloques no mejora las métricas. Por el contrario, induce inestabilidad severa en los gradientes (observada en las inmensas barras de error de los modelos 12, 13 y 16).
3. **Eficiencia Temprana:** Modelos ultra-superficiales (2 y 3 bloques) logran un rendimiento sorprendentemente competitivo, demostrando la alta capacidad de representación espacial-espectral del módulo DFHM incluso con parámetros mínimos.

## 📂 Estructura del Código
* `ELWRYM_net.py`: Arquitectura de la red neuronal optimizada.
* `dataset_cassi.py`: Dataloader y simulador físico de la dispersión CASSI.
* `train.py`: Script de orquestación para el entrenamiento secuencial y automatizado de los 20 modelos.
* `ELWRYM_plot.py`: Analizador estadístico que procesa los archivos `.npz` para generar los gráficos de medias y desviaciones estándar.

## 🚀 Uso
Para replicar el estudio de ablación:
1. Ejecuta `python train.py` para entrenar iterativamente los modelos (generará la carpeta `ELWRYM_1a20`).
2. Ejecuta `python ELWRYM_plot.py` para extraer las métricas de convergencia y generar la gráfica de resultados estadísticos.