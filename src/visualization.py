import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.data_processing import preprocess_image

def plot_clahe_comparison(dataframe, img_dir, num_samples=5):
    print(f"\nGenerando visualización de {num_samples} imágenes con CLAHE...")
 
    samples = dataframe.sample(n=num_samples, random_state=42)
    
    plt.figure(figsize=(15, 6))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(img_dir, row['filename'])
        img = cv2.imread(img_path)

        if img is None:
            print(f"No se pudo cargar la imagen: {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_clahe = preprocess_image(img_rgb)
        
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Original\n{row['label']}")
        plt.axis('off')
        
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(img_clahe)
        plt.title("Con CLAHE")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    

def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    """Genera y guarda la Matriz de Confusión"""
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - Diagnóstico Ocular')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig('./models/confusion_matrix.png')
    plt.show()

def plot_roc_curves(y_true_categorical, y_pred_prob, class_names):
    """Genera y guarda las Curvas ROC para justificar el AUC"""
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_categorical[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC Multi-clase')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('./models/roc_curves.png')
    plt.show()