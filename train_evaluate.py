import os
# Silenciamos los avisos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import keras
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# Importamos nuestras piezas personalizadas
from src.dataset import ODIRDataGenerator
from src.model import create_odir_model  # Asegúrate de haber actualizado src/model.py con la versión EfficientNet
from src.visualization import plot_confusion_matrix, plot_roc_curves

# --- CONFIGURACIÓN Y RUTAS ---
TRAIN_DIR = './data/Training Images'
TRAIN_CSV = './data/train_split.csv'
TEST_CSV = './data/test_split.csv'
MODEL_PATH = './models/best_medical_model.weights.h5'
MODEL_TYPE = 'efficientnetb0'  # Cambia a 'resnet50' si quieres usar ResNet50

def main():
    print(f"--- 1. CARGANDO DATOS PARA {MODEL_TYPE.upper()} ---")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print("\n--- 2. INICIALIZANDO GENERADORES ---")
    # Nota: El target_size es (224, 224) por defecto en nuestro generador
    train_gen = ODIRDataGenerator(train_df, TRAIN_DIR, batch_size=32)
    test_gen = ODIRDataGenerator(test_df, TRAIN_DIR, batch_size=32, shuffle=False)
    class_names = list(train_gen.label_mapping.keys())

    print("\n--- 3. TRATAMIENTO DE DESBALANCEO (PESOS) ---")
    # Conversión limpia para evitar avisos de Pylance
    etiquetas_lista = train_df['label'].map(train_gen.label_mapping).tolist()
    y_train_num = np.array(etiquetas_lista, dtype=int)
    clases_unicas = np.unique(y_train_num)
    
    pesos = compute_class_weight(class_weight='balanced', classes=clases_unicas, y=y_train_num)
    class_weight_dict = dict(zip(clases_unicas, pesos))
    
    for c_num, peso in class_weight_dict.items():
        print(f" - {class_names[c_num]}: {peso:.2f}")

    print("\n--- 4. CONSTRUYENDO ARQUITECTURA DEL NOTEBOOK ---")
    # Usamos freeze_features=True al principio para no "romper" los pesos de ImageNet
    model = create_odir_model(model_type=MODEL_TYPE, num_classes=5, freeze_features=True)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    # Creamos carpeta de modelos si no existe
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # CALLBACKS AVANZADOS DEL NOTEBOOK
    callbacks = [
        # Guarda el mejor basado en accuracy de validación
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', save_weights_only=True),
        # Para si el error no baja en 8 épocas
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        # Baja el learning rate si se estanca (Cambio de marchas)
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    print("\n--- 5. ENTRENAMIENTO ---")
    EPOCHS = 30 
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    print("\n--- 6. EVALUACIÓN FINAL ---")
    # Cargamos los mejores pesos obtenidos
    model.load_weights(MODEL_PATH)

    # Predicciones
    y_pred_prob = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)

    # Etiquetas reales (concatenamos todos los batches del generador de test)
    y_true_categorical = np.concatenate([y for x, y in test_gen], axis=0)
    y_true_classes = np.argmax(y_true_categorical, axis=1)

    print("\n" + "="*30)
    print("   RESULTADOS FINALES")
    print("="*30)
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))

    # Generamos las gráficas profesionales
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
    plot_roc_curves(y_true_categorical, y_pred_prob, class_names)
    
    print("\n¡Proyecto ejecutado! Revisa la carpeta 'models/' para ver las gráficas.")

if __name__ == '__main__':
    main()