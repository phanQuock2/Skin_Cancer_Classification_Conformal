import numpy as np

# Tạo một lớp NonconformityScorer để tính toán nonconformity scores cho các dự đoán của mô hình. 
class NonconformityScorer:
    def __init__(self, model):
        self.model = model
        self.threshold = None  # Lưu ngưỡng để tái sử dụng

    def fit(self, X_cal, y_cal):
        # Tính nonconformity scores cho tập calibration
        y_indices = np.argmax(y_cal, axis=1)
        probs = self.model.predict(X_cal, verbose=0)  
        self.nc_scores = -np.array([p[y] for p, y in zip(probs, y_indices)])

    def calculate_threshold(self, significance):
        if self.nc_scores is None:
            raise ValueError("Must call fit() before calculating threshold.")
        q = np.quantile(self.nc_scores, 1 - significance)
        self.threshold = q
        return q
    
class InductiveConformalPredictor:
    def __init__(self, model, calibration_fraction=0.2, batch_size=32):
        self.model = model
        self.cal_fraction = calibration_fraction
        self.nc_scorer = NonconformityScorer(model)
        self.n_classes = None
        self.batch_size = batch_size  # Thêm batch_size để kiểm soát kích thước batch

    def fit(self, X_cal, y_cal, significance = 0.05):
        # Huấn luyện với tập calibration và tính điểm không phù hợp
        self.nc_scorer.fit(X_cal, y_cal)
        self.n_classes = y_cal.shape[1]
        self.nc_scorer.calculate_threshold(significance)


    def predict_batch(self, X_batch, significance):
        # Tính dự đoán xác suất sử dụng batch prediction
        probas = self.model.predict(X_batch, verbose=0)  # Tắt verbose
        
        # Lấy hoặc tính ngưỡng nếu chưa có
        if self.nc_scorer.threshold is None or self.nc_scorer.threshold < 0:
            threshold = self.nc_scorer.calculate_threshold(significance)
        else:
            threshold = self.nc_scorer.threshold
        
        # Dùng numpy để tạo prediction sets nhanh hơn
        # Tạo mask cho các lớp có nonconformity score dưới ngưỡng
        neg_probs = -probas  # Nonconformity scores
        mask = neg_probs <= threshold
        
        # Chuyển đổi mask thành danh sách các lớp cho mỗi mẫu
        prediction_sets = [np.where(mask[i])[0].tolist() for i in range(len(mask))]
        
        return prediction_sets


