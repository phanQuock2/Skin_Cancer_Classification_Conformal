from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from conformal_utils import InductiveConformalPredictor, NonconformityScorer 

# Khởi tạo Flask 
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Define Top2 và top3 accuracy
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# Tải model gốc
model = load_model('model.h5', custom_objects={'top_2_accuracy': top_2_accuracy,'top_3_accuracy': top_3_accuracy})

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] 

label_to_vietnamese = {
    'nv': 'U hắc tố lành tính',
    'mel': 'U hắc tố ác tính',
    'bkl': 'Dày sừng lành tính',
    'bcc': 'Ung thư biểu mô tế bào đáy',
    'akiec': 'Dày sừng quang hóa / UTBM tế bào vảy tại chỗ',
    'vasc': 'Tổn thương da mạch máu',
    'df': 'U xơ da'
}


# Load ICP dữ liệu từ .npz
icp_data = np.load("icp_data.npz", allow_pickle=True)
scores = icp_data['scores']
threshold = float(icp_data['threshold'])
n_classes = int(icp_data['n_classes'])

# Tạo ICP instance và gán dữ liệu đã lưu
nc_scorer = NonconformityScorer(model)
nc_scorer.nc_scores = scores
nc_scorer.threshold = threshold

icp = InductiveConformalPredictor(model)
icp.nc_scorer = nc_scorer
icp.n_classes = n_classes

# Hàm xử lý ảnh
def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Xử lý upload
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('predict', filename=filename))
    return redirect('/')

# Dự đoán và hiển thị
@app.route('/predict/<filename>')
def predict(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  
    image_path = url_for('static', filename='uploads/' + filename)
    X, _ = prepare_image(img_path)

    # Dự đoán xác suất bằng model
    probs = model.predict(X, verbose=0)[0] 
    pred_probs = {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}

    # Dự đoán tập nhãn đáng tin cậy bằng ICP (prediction set)
    prediction_set_indices = icp.predict_batch(X, significance=0.1)[0]  # ví dụ: [0, 4, 6]
    prediction_set = {class_labels[i]: pred_probs[class_labels[i]] for i in prediction_set_indices}

    # Lấy Top 3 nhãn từ prediction set (xếp theo xác suất)
    top3_raw = sorted(prediction_set.items(), key=lambda x: x[1], reverse=True)
    top3 = [(label_to_vietnamese[label.lower()], prob) for label, prob in top3_raw]


    return render_template('result.html',
                           image_path=image_path,
                           top3_probs=top3)

# Khởi chạy
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
