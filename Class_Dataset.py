import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class MyDataset(Dataset): #class quản lí scalogram images
    def __init__(self, root_dir, transform = None):
        # Args:
        # root_dir (string): Đường dẫn đến thư mục chứa ảnh (ví dụ: 'data/train' hoặc 'data/test')
        self.root_dir = root_dir
        self.transform = transform

        # Lưu danh sách tất cả đường dẫn ảnh và label tương ứng
        self.image_paths = []
        self.labels = []
        self.rr_paths = []

        # Duyệt qua từng thư mục label (0, 1, 2, 3)
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label) #tạo đg dẫn tới thư mục từng label
            if os.path.isdir(label_dir): #kiểm tra xem label_dir có phải là một thư mục hay không (đúng thì TRUE)
                for file in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file)
                    
                    # Collect image and corresponding RR feature paths
                    if file.endswith('.png'):
                        base_name = file[:-4]  # Remove '.png' extension
                        rr_file = f"{base_name}_rr.npy"
                        rr_path = os.path.join(label_dir, rr_file)
                        
                        if os.path.exists(rr_path):  # Only add if RR features exist
                            self.image_paths.append(file_path)
                            self.rr_paths.append(rr_path)
                            self.labels.append(int(label))
    
    def __len__(self):
        #mỗi Dataset cần phải định nghĩa phương thức __len__() để DataLoader biết có bao nhiêu sample.
        #self.image_paths là một danh sách chứa đường dẫn đến tất cả các ảnh ECG bạn đã load vào trong __init__()
        return len(self.image_paths)    #trả về số lượng mẫu (số ảnh ECG) trong dataset
    
    def __getitem__(self, idx):
        # Mở ảnh
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # nếu muốn convert RGB thì image = Image.open(img_path).convert('RGB')

        #get label
        label = self.labels[idx]

        # Load RR features
        rr_path = self.rr_paths[idx]
        rr_features = np.load(rr_path)

        if self.transform:
            image = self.transform(image)

        # Convert RR features to tensor
        rr_features = torch.from_numpy(rr_features).float()

        return image, rr_features, label
    #    Returns:
    #         tuple: (image_tensor, rr_features, label) where:
    #             - image_tensor: transformed image tensor
    #             - rr_features: numpy array of RR interval features
    #             - label: class label