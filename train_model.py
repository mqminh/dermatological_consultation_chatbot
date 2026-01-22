import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# ==========================================
# 1. CẤU HÌNH THAM SỐ (CONFIGURATION)
# ==========================================
# Đổi đường dẫn này thành đường dẫn chứa folder dataset của bạn
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224)     # Kích thước chuẩn của MobileNetV2
BATCH_SIZE = 32           # Số lượng ảnh học trong 1 lần cập nhật trọng số
EPOCHS = 20               # Số vòng lặp huấn luyện (tùy chỉnh dựa trên độ chính xác)
LEARNING_RATE = 0.0001    # Tốc độ học (để thấp để tránh phá vỡ trọng số pre-trained)

# Kiểm tra GPU (Nếu có)
print(f"TensorFlow Version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ==========================================
# 2. LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU (DATA LOADING)
# ==========================================
print("\n--- Đang tải dữ liệu ---")

# Tự động chia Train (80%) và Validation (20%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Phát hiện {num_classes} loại bệnh: {class_names}")

# Tối ưu hóa hiệu năng load dữ liệu (Caching & Prefetching)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION)
# ==========================================
# Giúp model không học vẹt (overfitting) bằng cách xoay, lật ảnh ngẫu nhiên
data_augmentation = Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

# ==========================================
# 4. XÂY DỰNG MODEL (TRANSFER LEARNING)
# ==========================================
print("\n--- Đang xây dựng Model ---")

# Tải MobileNetV2 đã train trên ImageNet, bỏ lớp đầu ra (include_top=False)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# Đóng băng base_model để không train lại các đặc trưng cơ bản ban đầu
base_model.trainable = False

# Tạo kiến trúc model hoàn chỉnh
model = Sequential([
  layers.Input(shape=IMG_SIZE + (3,)),
  data_augmentation,                # Lớp tăng cường dữ liệu
  layers.Rescaling(1./127.5, offset=-1), # Chuẩn hóa pixel về khoảng [-1, 1]
  base_model,                       # MobileNetV2 core
  layers.GlobalAveragePooling2D(),  # Giảm chiều dữ liệu
  layers.Dropout(0.2),              # Tránh overfitting
  layers.Dense(num_classes, activation='softmax') # Lớp phân loại cuối cùng
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 5. HUẤN LUYỆN MODEL (TRAINING)
# ==========================================
print("\n--- Bắt đầu huấn luyện ---")

# Callback để lưu model tốt nhất trong quá trình train
checkpoint_path = "best_skin_model.keras"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Callback dừng sớm nếu model không tốt lên sau 5 epoch
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS,
  callbacks=[checkpoint, early_stopping]
)

# ==========================================
# 6. TRỰC QUAN HÓA KẾT QUẢ (VISUALIZATION)
# ==========================================
# Vẽ biểu đồ Accuracy và Loss để đưa vào báo cáo Đồ án
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Lưu biểu đồ
plt.savefig('training_history.png')
print("\n--- Đã lưu biểu đồ vào training_history.png ---")
print(f"--- Đã lưu model tốt nhất vào {checkpoint_path} ---")

# Lưu tên các class vào file text để load lại sau này
with open('class_names.txt', 'w') as f:
    for cls in class_names:
        f.write(f"{cls}\n")