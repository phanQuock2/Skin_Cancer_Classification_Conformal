import numpy as np

# Tạo một lớp NonconformityScorer để tính toán nonconformity scores cho các dự đoán của mô hình. 
class NonconformityScorer:
    def __init__(self, model):
        self.model = model
        self.nc_scores = None
        self.thresholds = {}  # support multiple thresholds

    def fit(self, X_cal, y_cal):
        y_indices = np.argmax(y_cal, axis=1)
        probs = self.model.predict(X_cal, verbose=0)
        self.nc_scores = -np.array([p[y] for p, y in zip(probs, y_indices)])

    def calculate_threshold(self, significance):
        if self.nc_scores is None:
            raise ValueError("Must call fit() before calculating threshold.")
        q = np.quantile(self.nc_scores, 1 - significance)
        self.thresholds[significance] = q
        return q
    
class InductiveConformalPredictor:
    def __init__(self, model, calibration_fraction=0.2, batch_size=32):
        self.model = model
        self.cal_fraction = calibration_fraction
        self.nc_scorer = NonconformityScorer(model)
        self.n_classes = None
        self.batch_size = batch_size

    def fit(self, X_cal, y_cal):
        self.nc_scorer.fit(X_cal, y_cal)
        self.n_classes = y_cal.shape[1]

    def predict_batch(self, X_batch, significance):
        probas = self.model.predict(X_batch, verbose=0)

        # Ensure threshold exists for the current significance
        if significance not in self.nc_scorer.thresholds:
            threshold = self.nc_scorer.calculate_threshold(significance)
        else:
            threshold = self.nc_scorer.thresholds[significance]

        neg_probs = -probas
        mask = neg_probs <= threshold
        prediction_sets = [np.where(mask[i])[0].tolist() for i in range(len(mask))]
        return prediction_sets


