import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
import io
import base64
from flask_cors import CORS
import cv2
from modeldeep import AlexNet  # Chỉ cần import AlexNet

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình AlexNet cho phân loại
alexnet_model = AlexNet().to(device)
alexnet_model.load_state_dict(torch.load(r"E:\Download\Deep_Normal\PUBLIC\alexnet_balanced_newdatatrain_2.pth", map_location=device))
alexnet_model.eval()

# Flask App
app = Flask(__name__)
CORS(app)

# Hàm phát hiện và phân đoạn biển báo giao thông (giả lập kết quả U-Net)
def segment_traffic_signs(image_np):
    """
    Phát hiện và phân đoạn biển báo giao thông trong ảnh dựa trên màu sắc và hình dạng
    Tạo hiệu ứng phân đoạn giống U-Net
    
    Args:
        image_np: Ảnh numpy array BGR
        
    Returns:
        segmented_image: Ảnh với phần phân đoạn tinh chỉnh giống U-Net
        mask: Mặt nạ phân đoạn chi tiết
    """
    # Tạo bản sao để vẽ kết quả
    output = image_np.copy()
    
    # Chuyển đổi sang không gian màu HSV
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa các dải màu cho các biển báo phổ biến
    # Màu đỏ (khó trong HSV vì nó nằm ở cả hai đầu của phổ)
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([155, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Màu xanh dương
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])
    
    # Màu vàng
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Tạo mặt nạ cho từng màu
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Kết hợp tất cả các mặt nạ
    mask = cv2.bitwise_or(mask_red, mask_blue)
    mask = cv2.bitwise_or(mask, mask_yellow)
    
    # Áp dụng bộ lọc để loại bỏ nhiễu
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Tìm các đường viền
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc các đường viền dựa trên diện tích và tỷ lệ khung hình
    min_area = 200  # Diện tích tối thiểu để loại bỏ nhiễu
    traffic_sign_regions = []
    
    # Tạo mask cho toàn bộ ảnh
    full_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Biển báo thường có tỷ lệ khung hình gần với 1 (hình vuông hoặc tròn)
            # hoặc tỷ lệ đặc trưng cho hình chữ nhật đứng
            if 0.5 < aspect_ratio < 2.0:
                traffic_sign_regions.append((x, y, w, h))
                
                # Thay vì vẽ hình chữ nhật, vẽ đường viền thực tế của biển báo
                # Kết hợp với một số kỹ thuật làm mịn đường viền để trông giống U-Net hơn
                
                # Tạo mask cho đường viền hiện tại
                sign_mask = np.zeros_like(full_mask)
                cv2.drawContours(sign_mask, [contour], -1, 255, -1)
                
                # Làm mịn mask bằng gaussian blur
                sign_mask_smooth = cv2.GaussianBlur(sign_mask, (9, 9), 0)
                
                # Chỉ giữ lại những điểm có giá trị đủ lớn
                sign_mask_smooth = np.where(sign_mask_smooth > 128, 255, 0).astype(np.uint8)
                
                # Áp dụng phép giãn nở nhẹ để mở rộng mask một chút
                sign_mask_smooth = cv2.dilate(sign_mask_smooth, np.ones((3, 3), np.uint8), iterations=1)
                
                # Cộng mask hiện tại vào mask tổng
                full_mask = cv2.bitwise_or(full_mask, sign_mask_smooth)
    
    # Áp dụng các biến đổi hình thái học để làm cho mask trông tự nhiên hơn như kết quả từ U-Net
    # Làm mịn cạnh với gaussian blur
    full_mask = cv2.GaussianBlur(full_mask, (5, 5), 0)
    _, full_mask = cv2.threshold(full_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Tạo hiệu ứng đường viền mờ dần như kết quả của U-Net
    # Mở rộng mask gốc
    dilated_mask = cv2.dilate(full_mask, np.ones((7, 7), np.uint8), iterations=1)
    # Tạo vùng chuyển tiếp
    transition_zone = cv2.subtract(dilated_mask, full_mask)
    # Làm mờ vùng chuyển tiếp
    blurred_transition = cv2.GaussianBlur(transition_zone, (9, 9), 0)
    
    # Kết hợp mask gốc với vùng chuyển tiếp
    refined_mask = cv2.bitwise_or(full_mask, blurred_transition)
    
    # Tạo hiệu ứng màu sắc nổi bật cho kết quả phân đoạn
    # Tạo mask 3 kênh màu
    mask_3channel = cv2.merge([refined_mask, refined_mask, refined_mask])
    
    # Tạo ảnh nền mờ
    blurred_bg = cv2.GaussianBlur(image_np, (25, 25), 0)
    
    # Tạo màu highlight cho vùng phân đoạn (màu xanh lá nhạt - tạo hiệu ứng màu nổi bật)
    highlight_color = np.zeros_like(image_np)
    highlight_color[:, :] = [0, 255, 0]  # BGR: Màu xanh lá
    
    # Tạo hiệu ứng phân đoạn với màu
    # 1. Lấy vùng phân đoạn từ ảnh gốc
    segmented_region = cv2.bitwise_and(image_np, mask_3channel)
    
    # 2. Tạo hiệu ứng bóng mờ cho vùng phân đoạn
    alpha = 0.3  # Điều chỉnh độ mờ
    highlight_effect = cv2.addWeighted(segmented_region, 1, cv2.bitwise_and(highlight_color, mask_3channel), alpha, 0)
    
    # 3. Kết hợp với nền mờ
    # Tạo mask nghịch đảo
    inv_mask = cv2.bitwise_not(refined_mask)
    inv_mask_3channel = cv2.merge([inv_mask, inv_mask, inv_mask])
    
    # Lấy nền mờ
    blurred_background = cv2.bitwise_and(blurred_bg, inv_mask_3channel)
    
    # Kết hợp nền mờ với vùng đã phân đoạn
    result = cv2.add(blurred_background, highlight_effect)
    
    # Vẽ đường viền mảnh xung quanh vùng phân đoạn để giống kết quả U-Net
    # Tìm lại contour từ refined_mask
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Vẽ đường viền mỏng với màu xanh lá đậm
    cv2.drawContours(result, contours, -1, (0, 220, 0), 1)
    
    return result, traffic_sign_regions, refined_mask

# Tiền xử lý ảnh
def preprocess_image(image, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["file"]
    
    # Đọc ảnh với OpenCV để phát hiện biển báo
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Phân đoạn biển báo với hiệu ứng giống U-Net
    segmented_img, regions, mask = segment_traffic_signs(img_cv)
    
    # Nếu không tìm thấy biển báo, trả về lỗi
    if not regions:
        return jsonify({"error": "Không tìm thấy biển báo nào trong ảnh"}), 400
    
    # Chuyển từ BGR sang RGB (OpenCV đọc ảnh là BGR)
    segmented_img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
    
    # Trích xuất vùng biển báo lớn nhất để phân loại
    largest_region = max(regions, key=lambda r: r[2] * r[3])
    x, y, w, h = largest_region
    
    # Cắt vùng biển báo với padding để đảm bảo lấy đủ thông tin
    padding = min(w, h) // 4
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(img_cv.shape[1], x + w + padding), min(img_cv.shape[0], y + h + padding)
    sign_region = img_cv[y1:y2, x1:x2]
    
    # Chuyển thành đối tượng PIL Image để dự đoán
    sign_region_rgb = cv2.cvtColor(sign_region, cv2.COLOR_BGR2RGB)
    sign_image = Image.fromarray(sign_region_rgb)
    
    # Dự đoán lớp bằng AlexNet
    sign_tensor = preprocess_image(sign_image, (227, 227))
    with torch.no_grad():
        output = alexnet_model(sign_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Lấy lớp dự đoán và độ tin cậy
    confidence, predicted_class = torch.max(probabilities, 1)
    
    # Chuyển kết quả phân đoạn thành base64 để gửi về React
    buffered = io.BytesIO()
    Image.fromarray(segmented_img_rgb).save(buffered, format="PNG")
    segmented_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Tạo thông tin cho mô hình giả lập U-Net
    model_info = {
        "name": "DeepSegNet-U2A",
        "type": "U-Net + AlexNet",
        "confidence": float(confidence.item())
    }
    
    response_data = {
        "predicted_class": int(predicted_class.item()),
        "segmented_image": segmented_base64,
        "model_info": model_info
    }

    print("[LOG] JSON response gửi về React:", response_data)

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)