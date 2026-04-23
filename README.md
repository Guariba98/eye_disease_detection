# Detección de Patologías Oculares (ODIR Dataset)

Este proyecto desarrolla un sistema de clasificación automática de imágenes de fondo de ojo (retinografías) utilizando Deep Learning. El modelo es capaz de identificar 5 categorías clínicas distintas basándose en el dataset ODIR (Ocular Disease Intelligent Recognition).

## Resumen del Proyecto

El objetivo es clasificar imágenes médicas en las siguientes categorías:
- **Normal** (Sano)
- **Cataract** (Cataratas)
- **Glaucoma**
- **Myopia** (Miopía)
- **Diabetes** (Retinopatía Diabética)

### Resultados Destacados
- **Accuracy Global:** 75%
- **F1-Score Myopia:** 0.92 (Excelente capacidad de detección)
- **F1-Score Cataract:** 0.83 (Alta fiabilidad)
- **Desafío superado:** Gestión de clases altamente desbalanceadas (ej. Diabetes con solo 16 muestras en el conjunto de entrenamiento).

## Tecnologías Utilizadas

- **Lenguaje:** Python 3.x
- **Deep Learning:** TensorFlow / Keras 3
- **Arquitectura:** EfficientNetB0 (Transfer Learning)
- **Procesamiento de Imagen:** OpenCV (CLAHE para mejora de contraste)
- **Análisis de Datos:** Pandas, NumPy, Scikit-learn
- **Visualización:** Matplotlib, Seaborn

## Estructura del Proyecto

```text
eye_disease_detection/
├── data/                   # Archivos CSV de entrenamiento/test y etiquetas
├── src/                    # Código fuente modular
│   ├── data_processing.py  # Limpieza y CLAHE
│   ├── dataset.py          # Generador de datos (Data Pipeline)
│   ├── model.py            # Arquitectura EfficientNetB0
│   └── visualization.py    # Gráficas de métricas y comparativas
├── 01_prepare_data.py      # Script de preparación y balanceo
├── 02_train_evaluate.py    # Script de entrenamiento y evaluación
└── README.md
