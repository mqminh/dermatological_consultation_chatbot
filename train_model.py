import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from sklearn.utils import class_weight
import numpy as np

# ==========================================
# 1. CẤU HÌNH TỐI ƯU (HYPERPARAMETERS)
# ==========================================
DATA_DIR = 'dataset'
IMG_SIZE = (300, 300)  # EfficientNetB3 tối ưu ở kích thước 300x300
BATCH_SIZE = 16        # Giảm batch size để vừa VRAM 8GB (vì ảnh to hơn)
EPOCHS_HEAD = 10       # Số epoch train khởi động
EPOCHS_FINE = 50       # Số epoch train tinh chỉnh (sẽ dừng sớm nếu cần)

# Kiểm tra GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Đang chạy trên GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)

# ==========================================
# 2. LOAD DATASET CHUẨN
# ==========================================
print("\n--- Đang tải và xử lý dữ liệu ---")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int' # Dùng sparse categorical crossentropy cho tiết kiệm RAM
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")
# ==========================================
# 2.5. TÍNH TOÁN CLASS WEIGHTS (FIX LỖI JSON)
# ==========================================
print("\n--- Đang tính toán Class Weights ---")

# Lấy nhãn từ tập train để tính toán
# Lưu ý: train_ds là dạng Batch, cần nối lại
train_labels = []
for images, labels in train_ds.unbatch():
    train_labels.append(labels.numpy())

train_labels = np.array(train_labels)

# Tính toán trọng số
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# QUAN TRỌNG: Chuyển về float thuần của Python để tránh lỗi JSON Serialized
class_weights_dict = {i : float(w) for i, w in enumerate(class_weights)}

print("Class Weights (Đã fix lỗi):")
print(class_weights_dict)

# Tối ưu hiệu năng pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. DATA AUGMENTATION NÂNG CAO
# ==========================================
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),       # Xoay tối đa 20%
    layers.RandomZoom(0.2),           # Zoom
    layers.RandomContrast(0.2),       # Thay đổi tương phản (quan trọng cho da liễu)
    layers.RandomBrightness(0.2),     # Thay đổi độ sáng
])

# ==========================================
# 4. XÂY DỰNG MODEL (EFFICIENTNET B3)
# ==========================================
print("\n--- Khởi tạo EfficientNetB3 ---")

# Tải base model
base_model = tf.keras.applications.EfficientNetB3(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# Ban đầu đóng băng toàn bộ base
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)

# EfficientNet có sẵn lớp xử lý input, nhưng ta dùng preprocess_input cho chắc chắn nếu cần
# x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False) # training=False để giữ nguyên BatchNormalization
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x) # Giúp ổn định training
x = layers.Dropout(0.3)(x)         # Tăng dropout lên 0.3 để chống overfitting
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# ==========================================
# 5. CÁC CALLBACKS QUAN TRỌNG
# ==========================================
# Lưu model tốt nhất (không phải model cuối cùng)
checkpoint = ModelCheckpoint(
    "best_skin_model_v2.h5",
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

# Dừng train nếu không tiến bộ sau 7 epoch
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# Giảm Learning Rate nếu Loss đi ngang (giúp hội tụ sâu hơn)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,    # Giảm 5 lần (nhân 0.2)
    patience=3,    # Sau 3 epoch không khá hơn thì giảm
    min_lr=1e-6,
    verbose=1
)

# ==========================================
# 6. GIAI ĐOẠN 1: WARM-UP (TRAIN HEAD)
# ==========================================
print("\n🔥 GIAI ĐOẠN 1: Train lớp Classifier (Warm-up)...")
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3), # LR ban đầu lớn
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[checkpoint] # Chỉ lưu checkpoint, chưa cần giảm LR
)

# ==========================================
# 7. GIAI ĐOẠN 2: FINE-TUNING TOÀN BỘ
# ==========================================
print("\n🔥🔥 GIAI ĐOẠN 2: Unfreeze toàn bộ và Train sâu...")

# Mở khóa toàn bộ model để học các đặc trưng chi tiết của da
base_model.trainable = True

# Quan trọng: Khi fine-tune phải dùng Learning Rate RẤT NHỎ
# Nếu không sẽ phá hỏng các trọng số đã học ở ImageNet
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4), # Nhỏ hơn 10 lần
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Nối tiếp history
total_epochs = EPOCHS_HEAD + EPOCHS_FINE

history_2 = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=history_1.epoch[-1],
    epochs=total_epochs,
    callbacks=[checkpoint, early_stopping, lr_scheduler], # Thêm đầy đủ "vũ khí"
    class_weight=class_weights_dict
)

# ==========================================
# 8. VẼ BIỂU ĐỒ BÁO CÁO
# ==========================================
acc = history_1.history['accuracy'] + history_2.history['accuracy']
val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
loss = history_1.history['loss'] + history_2.history['loss']
val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
# Vẽ đường ngăn cách 2 giai đoạn
plt.axvline(x=EPOCHS_HEAD-1, color='green', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Training Accuracy (EfficientNetB3)')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(x=EPOCHS_HEAD-1, color='green', linestyle='--')
plt.legend(loc='upper right')
plt.title('Training Loss')

plt.savefig('training_result_optimized.png')
print("\n✅ Đã hoàn tất! Model lưu tại: best_skin_model_v2.keras")
print("Biểu đồ kết quả: training_result_optimized.png")

# Lưu lại class names
with open('class_names.txt', 'w') as f:
    for cls in class_names:
        f.write(f"{cls}\n")