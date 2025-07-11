import React, { useState, useRef, useEffect } from "react";
import "../sakura-styles.css";
import { Link } from 'react-router-dom';
const API_URL = process.env.REACT_APP_API_URL;

const ImagePredictor = () => {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [segmentedImage, setSegmentedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [processingStage, setProcessingStage] = useState("");

  // Hiệu ứng tiến trình
  const [progress, setProgress] = useState(0);
  const progressTimer = useRef(null);

  // Tham chiếu để hiển thị kết quả
  const resultRef = useRef(null);

  // Hiệu ứng Processing
  useEffect(() => {
    if (isLoading) {
      setProgress(0);
      const stages = [
        "Đang tải mô hình U-Net...",
        "Tiền xử lý ảnh...",
        "Trích xuất đặc trưng...",
        "Thực hiện phân đoạn sâu...",
        "Tinh chỉnh kết quả...",
        "Tạo mặt nạ phân đoạn...",
        "Phân loại với mô hình AlexNet..."
      ];

      let currentStage = 0;
      setProcessingStage(stages[currentStage]);

      progressTimer.current = setInterval(() => {
        setProgress((prevProgress) => {
          const newProgress = prevProgress + (Math.random() * 2);
          
          if (newProgress >= 95) {
            clearInterval(progressTimer.current);
            return 95;
          }
          
          // Thay đổi giai đoạn xử lý
          if (newProgress > (currentStage + 1) * (100 / stages.length)) {
            currentStage = Math.min(currentStage + 1, stages.length - 1);
            setProcessingStage(stages[currentStage]);
          }
          
          return newProgress;
        });
      }, 200);
    } else {
      setProgress(100);
      clearInterval(progressTimer.current);
    }

    return () => {
      clearInterval(progressTimer.current);
    };
  }, [isLoading]);

  // Tự động cuộn xuống kết quả khi có dữ liệu
  useEffect(() => {
    if (segmentedImage && !isLoading && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [segmentedImage, isLoading]);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setSegmentedImage(null);
      setPrediction(null);
      setModelInfo(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!image) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    try {
      // Thêm độ trễ giả để làm cho hiệu ứng xử lý tốt hơn
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });;

      const textResponse = await response.text();
      console.log("[LOG] Raw response từ Flask:", textResponse);

      if (!response.ok) {
        throw new Error(`Server lỗi: ${response.status}`);
      }

      const data = JSON.parse(textResponse);
      console.log("[LOG] JSON response đã parse:", data);

      if (data.segmented_image && data.predicted_class !== undefined) {
        // Thêm độ trễ giả để hiệu ứng hoàn thành
        await new Promise(resolve => setTimeout(resolve, 500));
        
        setSegmentedImage(`data:image/png;base64,${data.segmented_image}`);
        setPrediction(data.predicted_class);
        setModelInfo(data.model_info || null);
        setError(null);
      } else {
        throw new Error("Dữ liệu phản hồi không hợp lệ!");
      }
    } catch (error) {
      console.error("Lỗi khi gọi API:", error);
      setError("Lỗi kết nối hoặc dữ liệu không hợp lệ!");
      setPrediction(null);
      setSegmentedImage(null);
      setModelInfo(null);
    } finally {
      setIsLoading(false);
    }
  };

  const getClassLabel = (classId) => {
    const labels = {
      0: "Nhãn 0: Biển báo 20km/h",
      1: "Nhãn 1: Biển báo 30km/h",
      2: "Nhãn 2: Biển báo 50km/h",
      3: "Nhãn 3: Biển báo 60km/h",
      4: "Nhãn 4: Biển báo 70km/h",
    };
    
    return labels[classId] || `Biển báo loại ${classId}`;
  };

  return (
    <div className="sakura-background">
      <div className="container">
        {/* Header */}
        <div className="header">
          <div className="cherry-blossom-icon"></div>
          <h1 className="main-title">DeepSegNet Traffic</h1>
          <p className="subtitle">
            Phân đoạn biển báo giao thông thông minh với mô hình U-Net
          </p>
          <Link to="/" className="back-button">
            <i className="fas fa-arrow-left"></i> Trang chủ
          </Link>
        </div>

        {/* Main Content */}
        <div className="main-content">
          {/* Left Column - Upload Section */}
          <div className="card upload-card">
            <h2 className="card-title">
              <i className="fas fa-cloud-upload-alt"></i>
              Tải lên ảnh
            </h2>

            <div className="upload-container">
              <label htmlFor="image-upload" className="upload-zone">
                {!preview ? (
                  <>
                    <i className="fas fa-image"></i>
                    <p>Kéo và thả hoặc nhấp để tải lên</p>
                    <span className="upload-hint">Hỗ trợ ảnh JPG, PNG</span>
                  </>
                ) : (
                  <p>Thay đổi ảnh khác</p>
                )}
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  className="hidden-input"
                />
              </label>
            </div>

            {preview && (
              <div className="preview-container">
                <div className="preview-box">
                  <img
                    src={preview}
                    alt="Uploaded"
                    className="preview-image"
                  />
                  <button
                    onClick={() => {
                      setPreview(null);
                      setImage(null);
                      setSegmentedImage(null);
                      setPrediction(null);
                      setModelInfo(null);
                      setError(null);
                    }}
                    className="remove-btn"
                  >
                    &times;
                  </button>
                </div>
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={!image || isLoading}
              className={`predict-btn ${(!image || isLoading) ? "disabled" : ""}`}
            >
              {isLoading ? (
                <span>
                  <i className="fas fa-circle-notch fa-spin"></i>
                  Đang xử lý...
                </span>
              ) : (
                <>
                  <i className="fas fa-microscope"></i>
                  Phân đoạn & Phân loại
                </>
              )}
            </button>

            {isLoading && (
              <div className="processing-indicator">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="processing-stage">{processingStage}</p>
              </div>
            )}

            {error && (
              <div className="error-message">
                <i className="fas fa-exclamation-triangle"></i>
                <p>{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Results Section */}
          <div className="card result-card" ref={resultRef}>
            <h2 className="card-title">
              <i className="fas fa-layer-group"></i>
              Kết quả phân đoạn
            </h2>

            {!segmentedImage && !prediction && !isLoading ? (
              <div className="empty-state">
                <div className="cherry-blossom-small"></div>
                <p>
                  Tải lên ảnh để xem kết quả phân đoạn U-Net và phân loại biển báo
                </p>
              </div>
            ) : (
              <>
                {isLoading && (
                  <div className="loading-state">
                    <i className="fas fa-circle-notch fa-spin"></i>
                    <p>Đang thực hiện phân đoạn sâu...</p>
                  </div>
                )}

                {prediction !== null && !isLoading && (
                  <div className="result-box">
                    <h3>Phân loại biển báo:</h3>
                    <div className="prediction-area">
                      <div className="prediction-badge">
                        {getClassLabel(prediction)}
                      </div>
                      {modelInfo && (
                        <div className="confidence-badge">
                          Độ tin cậy: {(modelInfo.confidence * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {segmentedImage && !isLoading && (
                  <div className="result-box">
                    <h3>Ảnh phân đoạn:</h3>
                    <div className="segmented-image-container">
                      <img
                        src={segmentedImage}
                        alt="Segmented"
                        className="segmented-image"
                      />
                    </div>
                    <p className="segmentation-info">
                      <i className="fas fa-info-circle"></i>
                      Kết quả phân đoạn sâu với mô hình U-Net
                    </p>
                  </div>
                )}

                {modelInfo && !isLoading && (
                  <div className="model-info-box">
                    <h4>Thông tin mô hình:</h4>
                    <div className="model-detail">
                      <span className="model-label">Mô hình:</span>
                      <span className="model-value">{modelInfo.name}</span>
                    </div>
                    <div className="model-detail">
                      <span className="model-label">Kiến trúc:</span>
                      <span className="model-value">{modelInfo.type}</span>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="footer">
          <p className="jp-footer">DeepSegNet Traffic - Phân đoạn biển báo giao thông với U-Net</p>
        </div>
      </div>
    </div>
  );
};

export default ImagePredictor;