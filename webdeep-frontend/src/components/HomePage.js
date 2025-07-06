import React from "react";
import { useNavigate } from "react-router-dom";
import "../sakura-styles.css";

const HomePage = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate("/predict");
  };

  return (
    <div className="home-gradient-background">
      <div className="home-container">
        <header className="home-header">
          <div className="logo-container">
            <div className="app-logo"></div>
            <h1 className="app-title">DEEP LEARNING</h1>
          </div>
          <nav className="main-nav">
            <ul>
              <li><a href="#features">Tính năng</a></li>
              <li><a href="#about">Giới thiệu</a></li>
              <li><button onClick={handleGetStarted} className="nav-cta">Bắt đầu</button></li>
            </ul>
          </nav>
        </header>

        <main>
          <section className="hero-section">
            <div className="hero-content">
              <h1 className="hero-title">Phân tích hình ảnh thông minh</h1>
              <p className="hero-subtitle">
                Sử dụng trí tuệ nhân tạo để phân đoạn và phân loại hình ảnh của bạn
                với độ chính xác cao
              </p>
              <div className="cta-buttons">
                <button onClick={handleGetStarted} className="primary-button">
                  <i className="fas fa-magic"></i>
                  Dự đoán ngay
                </button>
                <a href="#features" className="secondary-button">
                  <i className="fas fa-info-circle"></i>
                  Tìm hiểu thêm
                </a>
              </div>
            </div>
            <div className="hero-image">
              <div className="image-placeholder">
                <div className="image-effect"></div>
                <div className="image-preview"></div>
                <div className="image-segmented"></div>
              </div>
            </div>
          </section>

          <section id="features" className="features-section">
            <h2 className="section-title">Những tính năng nổi bật</h2>
            <div className="features-grid">
              <div className="feature-card">
                <div className="feature-icon upload-icon"></div>
                <h3>Tải lên dễ dàng</h3>
                <p>Kéo thả hoặc chọn hình ảnh của bạn một cách nhanh chóng và đơn giản</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon ai-icon"></div>
                <h3>AI phân tích</h3>
                <p>Thuật toán trí tuệ nhân tạo tiên tiến giúp phân tích hình ảnh chính xác</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon segment-icon"></div>
                <h3>Phân đoạn hình ảnh</h3>
                <p>Tự động phân đoạn các đối tượng trong hình ảnh với độ chính xác cao</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon result-icon"></div>
                <h3>Kết quả chi tiết</h3>
                <p>Nhận kết quả phân loại rõ ràng cùng với hình ảnh đã được xử lý</p>
              </div>
            </div>
          </section>

          <section id="about" className="about-section">
            <div className="about-content">
              <h2 className="section-title">NHÓM 6</h2>
              <p>
              <b>Bài toán: Đồ án tập trung vào việc xây dựng một hệ thống nhận dạng 5 loại biển báo giao thông bằng Deep Learning. Hệ thống bao gồm hai nhiệm vụ chính: </b><br/>
              •	Phân đoạn (Segmentation): Xác định vùng chứa biển báo trong ảnh. <br/>
              •	Phân loại (Classification): Nhận diện loại biển báo từ vùng đã phát hiện.
              </p>
              <p>
              <b>Phương pháp đề xuất giải quyết (method): </b><br/>
              •	Phân đoạn ảnh: Sử dụng mô hình U-Net, một kiến trúc mạnh mẽ cho bài toán phân vùng ảnh, giúp xác định chính xác vị trí biển báo trong ảnh có nhiều đối tượng. <br/>
              •	Phân loại biển báo: Sử dụng AlexNet, một mạng CNN sâu, để nhận diện chính xác từng loại biển báo sau khi đã phân đoạn. <br/>
              •	Dữ liệu: Ảnh biển báo được thu thập và tiền xử lý (resize, normalize, augmentation) để đảm bảo tính đa dạng. <br/>
              •	Huấn luyện: Áp dụng kỹ thuật tối ưu hóa (Adam optimizer), hàm mất mát phù hợp (Cross-Entropy Loss), và điều chỉnh siêu tham số để đạt hiệu suất tốt nhất.
              </p>
              <button onClick={handleGetStarted} className="primary-button">
                Trải nghiệm ngay
              </button>
            </div>
            <div className="about-image"></div>
          </section>
        </main>

        <footer className="home-footer">
          <div className="footer-content">
            <p>© 2025 DEEP LEARNING - Mô hình nhận diện biển báo giao thông</p>
            <div className="footer-links">
              <a href="#">Điều khoản sử dụng</a>
              <a href="#">Chính sách bảo mật</a>
              <a href="#">Liên hệ</a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default HomePage;