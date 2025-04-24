import os
from tqdm import tqdm #tqdm dùng để hiện progress bar khi xử lý nhiều ảnh
import concurrent.futures
from functools import partial
import shutil
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Đường dẫn (sử dụng Path thay vì os.path cho nhất quán)
SOURCE_DIR = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_CWT_RRI_CNN/data/scalogram_and_rr_interval")
TRAIN_DIR = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_CWT_RRI_CNN/data/train_data")
TEST_DIR = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_CWT_RRI_CNN/data/test_data")
SPLIT_RATIO = 0.7  # 70% train, 30% test
RANDOM_STATE = 4  # Để đảm bảo reproducibility

def get_image_rr_pairs(source_dir):
    #Lấy danh sách các cặp (ảnh, rr_features) cùng label"""
    image_rr_pairs = []
    labels = []
    
    for label in os.listdir(source_dir):
        label_dir = source_dir / label
        if label_dir.is_dir():
            # Lấy tất cả file ảnh (bỏ qua file _rr.npy)
            image_files = [f for f in os.listdir(label_dir) if f.endswith('.png') and not f.endswith('_rr.npy')]
            
            for img_file in image_files:
                base_name = img_file[:-4]  # Bỏ đuôi .png
                rr_file = f"{base_name}_rr.npy"
                img_path = label_dir / img_file
                rr_path = label_dir / rr_file
                
                if rr_path.exists():  # Chỉ thêm nếu có cả 2 file
                    image_rr_pairs.append((img_path, rr_path))
                    labels.append(label)
    
    return image_rr_pairs, labels

def copy_files(src_pair, label, target_root):
    #Copy cả ảnh và file RR features sang thư mục đích"""
    img_src, rr_src = src_pair
    label_dir = target_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy ảnh
    shutil.copy(img_src, label_dir / img_src.name)
    # Copy RR features
    shutil.copy(rr_src, label_dir / rr_src.name)

def main():
    # Bước 1: Lấy danh sách các cặp (ảnh, rr_features) và nhãn
    pairs, labels = get_image_rr_pairs(SOURCE_DIR)
    print(f"Tổng số mẫu tìm thấy: {len(pairs)}")
    
    # Bước 2: Chia train-test theo tỉ lệ, stratified
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, 
        test_size=1-SPLIT_RATIO, 
        stratify=labels,
        random_state=RANDOM_STATE
    )
    
    # Bước 3: Tạo thư mục đích nếu chưa tồn tại
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bước 4: Copy song song
    print("📂 Đang copy dữ liệu train...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(
            executor.map(copy_files, train_pairs, train_labels, [TRAIN_DIR]*len(train_pairs)),
            total=len(train_pairs)
        ))
    
    print("📂 Đang copy dữ liệu test...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(
            executor.map(copy_files, test_pairs, test_labels, [TEST_DIR]*len(test_pairs)),
            total=len(test_pairs)
        ))
    
    print(f"\n✅ Hoàn thành! Train: {len(train_pairs)} mẫu | Test: {len(test_pairs)} mẫu")

if __name__ == "__main__":
    main()