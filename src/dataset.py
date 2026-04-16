import os
import cv2
import numpy as np
import keras
from src.data_processing import preprocess_image

class ODIRDataGenerator(keras.utils.Sequence):
    def __init__(self, dataframe, img_dir, batch_size=32, target_size=(224, 224), shuffle=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        
        # Mapeamos los textos a números para clasificación
        self.label_mapping = {
            'Normal': 0, 'Cataract': 1, 'Glaucoma': 2, 'Myopia': 3, 'Diabetes': 4
        }
        self.num_classes = len(self.label_mapping)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_df = self.dataframe.iloc[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_df)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __data_generation(self, batch_df):
        X = np.empty((len(batch_df), *self.target_size, 3), dtype=np.float32)
        y = np.empty((len(batch_df)), dtype=int)
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_path = os.path.join(self.img_dir, row['filename'])
            image = cv2.imread(img_path)
            
            if image is not None:
                # 1. Leer y redimensionar
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.target_size)
                
                # 2. Aplicar tu función CLAHE
                image = preprocess_image(image)
                
                X[i,] = image
                y[i] = self.label_mapping[row['label']]
                
        # Convertimos las etiquetas a formato "One-Hot"
        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)