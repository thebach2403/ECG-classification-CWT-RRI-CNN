import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch.backends import cudnn
import torch.nn as nn
import torch
import torch.optim as optim 
from pathlib import Path
from tqdm import tqdm

from Class_Dataset import MyDataset
from Class_Model import MyModule


DATA_ROOT_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_CWT_RRI_CNN/data")
OUTPUT_MODEL_PATH = Path("E:/pv/WORKING/ECG_main_folder/ECG_Classification_CWT_RRI_CNN/model_save/ecg_model.pth")
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Áp dụng các biến đổi cho ảnh (chuyển thành tensor và chuẩn hóa)
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Đảm bảo kích thước ảnh là 100x100
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
])
########################     EXECUTE   ###########################

# Khởi tạo dataset và dataloader
train_dataset = MyDataset(root_dir = DATA_ROOT_PATH/'train_data',transform=transform)
test_dataset = MyDataset(root_dir = DATA_ROOT_PATH/'test_data',transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True,num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False,num_workers=4) #file test ko nên shuffle

# khởi tạo mô hình
model = MyModule()

# Chọn device: sử dụng GPU nếu có, không thì sử dụng CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Cài đặt loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train mô hình
def main():
    for epoch in range(num_epochs):
        model.train()  # Đặt mô hình vào chế độ huấn luyện
        running_loss = 0.0
        correct = 0
        total = 0
        for images,rr_features, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            rr_features = rr_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images,rr_features)
            # Tính loss
            loss = criterion(outputs, labels)
            # Backward pass và tối ưu hóa
            optimizer.zero_grad()  # Đặt gradient về 0
            loss.backward()  # Tính gradient
            optimizer.step()  # Cập nhật trọng số
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # In kết quả mỗi epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}, Accuracy: {100 * correct/total:.2f}%")

if __name__ == '__main__':
    main()
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print("✅ Đã lưu mô hình vào ecg_model.pth")

