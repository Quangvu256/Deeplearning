# 🚦 Traffic Sign Detection Web App (Flask + React)

Ứng dụng web phát hiện & phân loại biển báo giao thông:
- **Phân đoạn:** Giả lập U-Net (OpenCV)
- **Phân loại:** AlexNet (PyTorch)

Triển khai fullstack:
- Backend: **Python Flask**
- Frontend: **React.js**

---

## 📦 Cấu trúc dự án
PUBLIC/
├── .venv/
├── __pycache__/
├── PreImg/
├── webdeep-frontend/        # Thư mục React (frontend)
├── alexnet_balanced_newdatatrain_2.pth
├── alexnet_newdatatrain.pth
├── best_alexnet_model_balanced.pth
├── best_unet_model.pth
├── modeldeep.py             # File định nghĩa model AlexNet
├── README.txt
├── run_web.bat
└── WEBDEEP.py               # Flask backend chính

---
## Clone đầy đủ repo (bao gồm file mô hình lớn)

```bash
# Bước 1: Cài Git LFS (chỉ cần 1 lần)
git lfs install

# Bước 2: Clone repo
git clone https://github.com/Quangvu256/Deeplearning.git

## ✅ Yêu cầu hệ thống
- Python >= 3.8
- Node.js (>= 16)
- npm
- (Khuyến nghị) Tạo môi trường ảo:  
```bash
python -m venv .venv
⚙️ Hướng dẫn cài đặt
pip install -r requirements.txt
cd webdeep-frontend
npm install
Nhấn 2 lần vào file: run_web.bat
🌐 Kết quả
Frontend React chạy ở: http://localhost:3000

Backend Flask API chạy ở: http://localhost:5000
Hoặc có thể demo giao diện qua web : https://deeplearning2.onrender.com


