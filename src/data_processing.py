import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split

def get_clean_label(diagnosis_text):
    text = str(diagnosis_text).lower() 

    if 'cataract' in text: return 'Cataract'
    if 'glaucoma' in text: return 'Glaucoma'
    if 'diabetes' in text or 'diabetic' in text: return 'Diabetes'
    if 'myopia' in text: return 'Myopia'
    if 'normal' in text: return 'Normal'

    return 'Other'

def preprocess_image(img):
    if img.dtype != 'uint8':
        img = img.astype('uint8')
    # Aplicamos CLAHE (tu lógica original)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # IMPORTANTE: Devolvemos los píxeles de 0 a 255
    return final_img.astype('float32')

def load_and_prepare_data(excel_path):
    print(f"Leyendo Excel desde: {excel_path}")
    df = pd.read_excel(excel_path)
    data_clean = []

    for _, row in df.iterrows():
        l_label = get_clean_label(row['Left-Diagnostic Keywords'])
        if l_label != 'Other':
            data_clean.append({
                'filename': row['Left-Fundus'],
                'label': l_label,
                'eye': 'Left'
            })

        r_label = get_clean_label(row['Right-Diagnostic Keywords'])
        if r_label != 'Other':
            data_clean.append({
                'filename': row['Right-Fundus'],
                'label': r_label,
                'eye': 'Right'
            })

    df_v2 = pd.DataFrame(data_clean)
    
    print(f"\n--- RESUMEN DEL DATASET ---")
    print(f"Total de imágenes útiles encontradas: {len(df_v2)}")
    print("\nDistribución de clases original:")
    print(df_v2['label'].value_counts())
    
    df_normal = df_v2[df_v2['label'] == 'Normal'].sample(n=500, random_state=42)
    df_rest = df_v2[df_v2['label'] != 'Normal']
    df_balanced = pd.concat([df_normal, df_rest]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n--- NUEVA DISTRIBUCIÓN EQUILIBRADA ---")
    print(df_balanced['label'].value_counts())
    
    train_df, test_df = train_test_split(df_balanced, test_size=0.2, stratify=df_balanced['label'], random_state=42)
    
    print(f"\nImágenes para Entrenar: {len(train_df)}")
    print(f"Imágenes para Testear: {len(test_df)}\n")
    
    return train_df, test_df