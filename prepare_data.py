import os
from src.data_processing import load_and_prepare_data
from src.visualization import plot_clahe_comparison

EXCEL_PATH = './data/data.xlsx'
TRAIN_DIR = './data/Training Images'
OUTPUT_DIR = './data'

def main():
    train_df, test_df = load_and_prepare_data(EXCEL_PATH)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_split.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_split.csv'), index=False)

    print("\n--- FASE 2: Visualización de CLAHE ---")
    plot_clahe_comparison(train_df, TRAIN_DIR, num_samples=5)

if __name__ == '__main__':
    main()