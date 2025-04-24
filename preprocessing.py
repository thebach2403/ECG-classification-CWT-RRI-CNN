import os
import numpy as np
import pywt
import wfdb
import scipy.signal as sg
from pathlib import Path
from glob import glob
import joblib
from tqdm import tqdm
import cv2
from tqdm import tqdm #tqdm is used to show a progress bar when processing many images
import concurrent.futures


###########################          GLOBAL VARIABLES           ################################################

ROOT_PATH = "E:/pv/WORKING/ECG_main_folder/ECG_Classification_CWT_RRI_CNN/data/raw/mit-bih-arrhythmia-database-1.0.0"
OUTPUT_PATH = "E:/pv/WORKING/ECG_main_folder/ECG_Classification_CWT_RRI_CNN/data/scalogram_and_rr_interval"
sampling_rate = 360
wavelet = "mexh"  # mexh, morl, gaus8, gaus4
scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

# List of all record files (.dat)
record_files = sorted(glob(os.path.join(ROOT_PATH, '*.dat')))

cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1  # for multi-process

############################          SUPPORT FUNCTIONS          ###############################################

def read_ecg_record(record):
    record_path = Path(ROOT_PATH)/ str(record) # Combine PATH and record using pathlib
    signal = wfdb.rdrecord(record_path.as_posix(), channels=[0]).p_signal[:, 0]
    annotation = wfdb.rdann(record_path.as_posix(), extension="atr")
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)
    return signal,r_peaks,labels

def median_filter(signal):
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    filtered_signal = signal - baseline 
    return filtered_signal

def remove_invalid_labels(r_peaks, labels):
    invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
        #enumerate(labels) sẽ tạo ra một danh sách các cặp (index, label)
        #labels = ['N', '|', 'V', '~'] → enumerate(labels) sinh ra (0, 'N'), (1, '|'), (2, 'V'), (3, '~')
        #Vòng lặp for i, label in enumerate(labels)
        #Điều kiện if label not in invalid_labels
        #  ==> Danh sách kết quả indices Chứa danh sách các chỉ số i của label hợp lệ
    r_peaks, labels = r_peaks[indices], labels[indices]
    return r_peaks, labels

def align_r_peaks(r_peaks, filtered_signal,tol=0.05):
    newR = []
        #danh sách rỗng newR để lưu vị trí đỉnh R đã được căn chỉnh
    for r_peak in r_peaks: #Lặp qua từng điểm đỉnh R (r_peak) đã phát hiện trước đó.
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
            #Xác định giới hạn trái: Lùi lại một khoảng tol * sampling_rate từ vị trí r_peak để tạo một cửa sổ tìm kiếm.
            #np.maximum(..., 0) để đảm bảo không bị âm (tránh lỗi khi r_peak ở đầu tín hiệu).
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
            #Xác định giới hạn phải
        newR.append(r_left + np.argmax(filtered_signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int") # ép kiểu newR về int 

    #normalize signal 
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])
    return r_peaks, normalized_signal

# Nhóm các label thành 5 class
def AAMI_categories(labels): 
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # Normal
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB Supraventricular ectopic
        "V": 2, "E": 2,  # VEB Ventricular ectopic
        "F": 3,  # F Fusion beats
        "/": 4, "f": 4,"Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]
        #Nếu label = "N", thì AAMI["N"] sẽ trả về 0
        #[AAMI[label] for label in labels] là list comprehension trong Python, nó tạo ra một list mới. 
        #Kết quả là một danh sách các giá trị mã phân loại tương ứng với từng nhãn trong labels.
    return categories

def segmentation (normalize_signal, r_peaks, categories):
    before, after = 90, 110 ## Lấy đoạn tín hiệu 200 sample quanh R-peak
    beats, beat_labels, beat_index = [], [], [] #beat index để lưu rr ở phía sau

    for index, (r, category) in enumerate(zip(r_peaks, categories)):
        start = r - before
        end = r + after

        if category!=4 and start >= 0 and end < len(normalize_signal) and index > 0 and index <len(r_peaks-1): 
            # bỏ Q:4,  nếu không đủ dữ liệu (gần biên) cũng bỏ qua
            # segmentation
            beat = normalize_signal[start:end] # 1 khoảng tín hiệu nhịp tim
            beats.append(beat)
            beat_labels.append(category)
            beat_index.append(index)
        
    return beats,beat_labels, beat_index

def compute_rr_features(r_peaks, beat_index):

    avg_rri = np.mean(np.diff(r_peaks))
    rr_features_list = []

    for idx in beat_index:
        RR_previous = r_peaks[idx] - r_peaks[idx - 1] - avg_rri
        RR_post = r_peaks[idx + 1] - r_peaks[idx] - avg_rri
        RR_ratio = (r_peaks[idx] - r_peaks[idx - 1]) / (r_peaks[idx + 1] - r_peaks[idx])
        local_start = max(idx - 10, 0)
        RR_local = np.mean(np.diff(r_peaks[local_start:idx + 1])) - avg_rri

        rr_features = np.array([RR_previous, RR_post, RR_ratio, RR_local])
        rr_features_list.append(rr_features)

    return rr_features_list

# lưu rr_feature vào hàm này luôn để đồng bộ tên vs ảnh cwt
def CWT(record_name, beats, beat_labels, rr_features_list, beat_index):

    image_size = 100
    os.makedirs(OUTPUT_PATH, exist_ok=True) #Tạo thư mục gốc để lưu ảnh (nếu chưa tồn tại). Dùng exist_ok=True để tránh lỗi nếu thư mục đã có
    
    for i, (beat, beat_label,rr, idx) in enumerate(tqdm(zip(beats, beat_labels,rr_features_list,beat_index), total=len(beats))):
            #zip(beats, beat_labels)	Gộp hai list beats và beat_labels lại thành các cặp (beat, label)
            #tqdm(...)	Hiển thị progress bar

        # Consistent base filename
        base_filename = f"{record_name}_{beat_label}_{idx}"

        # Chuyển CWT
        # 1. Tính CWT → scalogram
        coef, _ = pywt.cwt(beat, scales, wavelet)
        scalogram = np.abs(coef)
        # 2. Chuẩn hóa về [0, 255] để lưu ảnh
        scalogram -= np.min(scalogram)
        scalogram /= np.max(scalogram) + 1e-6 #+1e-6: tránh chia cho 0
        scalogram *= 255
        scalogram = scalogram.astype(np.uint8) #Đổi sang uint8: đúng định dạng ảnh grayscale.
        # 3. Resize ảnh
        resized = cv2.resize(scalogram, (image_size, image_size), interpolation=cv2.INTER_CUBIC) #dùng nội suy INTER_CUBIC để giữ chất lượng cao
        # 4. Tạo thư mục theo label
        label_dir = os.path.join(OUTPUT_PATH, str(beat_label))
        os.makedirs(label_dir, exist_ok=True)
        # 5. Lưu ảnh dưới dạng .png (grayscale)
        image_path = os.path.join(label_dir, f"{base_filename}.png")
        cv2.imwrite(image_path, resized)  # ảnh grayscale

        # 6. Lưu RR-feature dạng .npy
        rr_path = os.path.join(label_dir, f"{base_filename}_rr.npy")
        np.save(rr_path, rr)


##############################              PROCESS 1 RECORD                   ####################################
def process_record(record_name):

    print(f'\n Processing record: {record_name}')

    # 1. Load dữ liệu
    signal,r_peaks,labels = read_ecg_record(record_name) 
    # 2. Lọc tín hiệu
    filtered_signal = median_filter(signal)
    # 3. lọc nhãn invalid
    r_peaks, labels = remove_invalid_labels(r_peaks,labels)
    # 4. căn chỉnh R, chuẩn hóa
    r_peaks, normalize_signal = align_r_peaks(r_peaks,filtered_signal)
    # 5. đổi sang dạng AAMI
    categories = AAMI_categories(labels)
    # 6. segmentation và tính rr-intervals 
    beats, beat_labels, beat_index = segmentation(normalize_signal,r_peaks,categories)
    # 7. tính rr features
    rr_features_list = compute_rr_features(r_peaks, beat_index)
    # 8. Chuyển thành scalogram và lưu ( ĐÔNG THỜI LƯU RR)
    CWT(record_name,beats, beat_labels,rr_features_list,beat_index)
    print(f'done: {record_name}')

def main():
    all_records = [
            '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
            '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
            '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
            '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
            '231', '232', '233', '234'
        ]
    # Sử dụng ProcessPoolExecutor để chạy song song
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_record, all_records)

if __name__ == "__main__":
    main()
