# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import os
import pandas as pd

# def train_mobilenet_model(train_dir, val_dir, image_size=224, batch_size=32, epochs=20):
#     # Data augmentation & preprocessing
#     datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#     train_batches = datagen.flow_from_directory(
#         train_dir,
#         target_size=(image_size, image_size),
#         batch_size=batch_size,
#         class_mode='categorical'
#     )

#     valid_batches = datagen.flow_from_directory(
#         val_dir,
#         target_size=(image_size, image_size),
#         batch_size=batch_size,
#         class_mode='categorical'
#     )

#     os.makedirs("models", exist_ok=True)
#     model_path = "models/mobilenet_last.h5"
#     best_model_path = "models/mobilenet_best.h5"
#     log_path = "models/training_log.csv"

#     # Resume from last epoch if log exists
#     initial_epoch = 0
#     if os.path.exists(log_path):
#         log_df = pd.read_csv(log_path)
#         if not log_df.empty:
#             initial_epoch = log_df['epoch'].max() + 1
#             print(f"Tiếp tục từ epoch {initial_epoch}")

#     # Load or create model
#     if os.path.exists(model_path):
#         print("Đang tải lại mô hình đã lưu gần nhất...")
#         model = load_model(model_path)
#     else:
#         print("Tạo mô hình mới từ MobileNet...")
#         base_model = MobileNet(include_top=False, input_shape=(image_size, image_size, 3))
#         for layer in base_model.layers:
#             layer.trainable = False

#         x = base_model.output
#         x = GlobalAveragePooling2D()(x)
#         x = Dropout(0.25)(x)
#         predictions = Dense(7, activation='softmax')(x)

#         model = Model(inputs=base_model.input, outputs=predictions)
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Callbacks
#     callbacks = [
#         EarlyStopping(patience=5, restore_best_weights=True),
#         ReduceLROnPlateau(patience=3),
#         ModelCheckpoint(best_model_path, save_best_only=True, verbose=1),
#         ModelCheckpoint(model_path, save_best_only=False, verbose=1),
#         CSVLogger(log_path, append=True)
#     ]

#     # Training
#     model.fit(
#         train_batches,
#         validation_data=valid_batches,
#         epochs=epochs,
#         initial_epoch=initial_epoch,
#         callbacks=callbacks
#     )

#     print(f"✅ Đã huấn luyện xong. Mô hình tốt nhất lưu ở: {best_model_path}")
#     print(f"🕒 Có thể tiếp tục huấn luyện từ: {model_path}, epoch {initial_epoch}")


def get_model():
    # Tao base model
    base_model = tf.keras.applications.mobilenet.MobileNet()
    
    # Lay cac layer cuoi cung cua base model
    x = base_model[-6].output

    # Lam giam kich thuoc cua x
    x = GlobalAveragePooling2D()(x)

    # Dropout layer
    x = Dropout(0.25)(x)    

    # Dense layer voi activation la softmax
    predictions = Dense(7, activation='softmax')(x)

    # Dong bang cac layer cua base model
    for layer in base_model.layers[:-23]:
        layer.trainable = False

    # Tao model chinh
    model = Model(inputs=base_model.input, outputs=predictions)
    return model



    