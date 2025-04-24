import torch.nn.functional as F
from torch.backends import cudnn
import torch.nn as nn
import torch

#Cấu hình PyTorch
cudnn.benchmark = False
cudnn.deterministic = True #Đảm bảo rằng mô hình luôn tạo ra kết quả giống nhau khi huấn luyện lại.

torch.manual_seed(0)

#Xây dựng mô hình CNN
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        #conv1 → conv3: Ba lớp convolutional trích xuất đặc trưng từ ảnh scalogram
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        #bn1 → bn3: Batch normalization giúp mô hình hội tụ nhanh hơn
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        #pooling1 → pooling3: Giảm kích thước feature map
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        #fc1, fc2: Lớp fully connected, ánh xạ đặc trưng vào 4 lớp output
        self.fc1 = nn.Linear(68, 32)
        self.fc2 = nn.Linear(32, 4)
    
    #Quá trình forward:
    #Mô hình nhận vào 2 đầu vào: x1 (scalogram) và x2 (RR). Sau đó, chúng được kết hợp và truyền qua các lớp của mô hình
    def forward(self, x1, x2):
        #Trích xuất đặc trưng từ ảnh scalogram x1
        #Lớp tích chập 1 (conv1)
        #Đầu vào: (1, 100, 100) (1 kênh ảnh scalogram kích thước 100×100)
        #conv1: Biến đổi thành (16, 94, 94)
        x1 = F.relu(self.bn1(self.conv1(x1)))  # (16 x 94 x 94)
        #pooling1: Giảm kích thước xuống (16, 18, 18)
        x1 = self.pooling1(x1)  # (16 x 18 x 18)
        #conv2: Biến đổi thành (32, 16, 16)
        x1 = F.relu(self.bn2(self.conv2(x1)))  # (32 x 16 x 16)
        #pooling2: Giảm kích thước xuống (32, 5, 5)
        x1 = self.pooling2(x1)  # (32 x 5 x 5)
        #conv3: Biến đổi thành (64, 3, 3)
        x1 = F.relu(self.bn3(self.conv3(x1)))  # (64 x 3 x 3)
        #pooling3: Giảm kích thước xuống (64, 1, 1)
        x1 = self.pooling3(x1)  # (64 x 1 x 1)
        #Flatten dữ liệu từ (64, 1, 1) được biến đổi thành vector 64 chiều
        x1 = x1.view((-1, 64))  # (64,)
        #x1 (vector 64 ch) và x2 (vector 4 ch) được nối lại thành vector có 68 giá trị
        x = torch.cat((x1, x2), dim=1)  # (68,)
        
        #Ánh xạ vào lớp fully connected
        #fc1: Giảm số chiều từ 68 xuống 32
        x = F.relu(self.fc1(x))  # (32,)
        #fc2: Dự đoán xác suất cho 4 lớp
        x = self.fc2(x)  # (4,)
        return x
    

