import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps

# ==========================================
# CẤU HÌNH CHIẾN LƯỢC
# ==========================================
DATA_DIR = '../dataset'
BACKUP_DIR = 'dataset_backup_excess'  # Nơi chứa ảnh thừa (để dành)

# Ngưỡng vàng cho bài toán này
MIN_SAMPLES = 600  # Các class < 600 sẽ được nhân bản lên 600
MAX_SAMPLES = 1000  # Các class > 1000 sẽ bị cắt bớt xuống 1000


def fix_imbalance():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    print(f"--- BẮT ĐẦU CÂN BẰNG DỮ LIỆU ---")
    print(f"Chiến lược: Đưa tất cả về khoảng [{MIN_SAMPLES} - {MAX_SAMPLES}] ảnh/class\n")

    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)
        # Lấy tất cả ảnh hợp lệ
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        images = [f for f in os.listdir(class_path) if f.lower().endswith(valid_exts)]
        count = len(images)

        print(f"Xử lý class '{cls}' ({count} ảnh):")

        # TRƯỜNG HỢP 1: DƯ DỮ LIỆU (Undersampling)
        if count > MAX_SAMPLES:
            excess = count - MAX_SAMPLES
            print(f"  -> Dư {excess} ảnh. Đang di chuyển sang Backup...")

            # Tạo folder backup tương ứng
            backup_class_path = os.path.join(BACKUP_DIR, cls)
            if not os.path.exists(backup_class_path):
                os.makedirs(backup_class_path)

            # Chọn ngẫu nhiên ảnh để di chuyển đi
            files_to_move = random.sample(images, excess)
            for f in files_to_move:
                src = os.path.join(class_path, f)
                dst = os.path.join(backup_class_path, f)
                shutil.move(src, dst)
            print("  -> Đã cắt giảm xong.")

        # TRƯỜNG HỢP 2: THIẾU DỮ LIỆU (Oversampling)
        elif count < MIN_SAMPLES:
            needed = MIN_SAMPLES - count
            print(f"  -> Thiếu {needed} ảnh. Đang sinh thêm (Augmentation)...")

            # Lấy đường dẫn đầy đủ của các ảnh hiện có
            img_paths = [os.path.join(class_path, f) for f in images]

            for i in range(needed):
                # Chọn ngẫu nhiên 1 ảnh gốc
                src_img_path = random.choice(img_paths)

                try:
                    with Image.open(src_img_path) as img:
                        # Convert sang RGB để tránh lỗi kênh màu
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Áp dụng biến đổi ngẫu nhiên
                        aug_type = random.choice(['FLIP', 'ROTATE', 'BRIGHTNESS', 'ZOOM'])

                        if aug_type == 'FLIP':
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        elif aug_type == 'ROTATE':
                            img = img.rotate(random.randint(-20, 20))
                        elif aug_type == 'BRIGHTNESS':
                            enhancer = ImageEnhance.Brightness(img)
                            img = enhancer.enhance(random.uniform(0.8, 1.2))
                        elif aug_type == 'ZOOM':
                            # Crop nhẹ ở giữa rồi resize lại
                            w, h = img.size
                            zoom = random.uniform(0.85, 0.95)
                            new_w, new_h = int(w * zoom), int(h * zoom)
                            left = (w - new_w) / 2
                            top = (h - new_h) / 2
                            img = img.crop((left, top, left + new_w, top + new_h))
                            img = img.resize((w, h), Image.Resampling.LANCZOS)

                        # Lưu ảnh mới
                        base_name = os.path.basename(src_img_path)
                        name, ext = os.path.splitext(base_name)
                        new_name = f"{name}_aug_{i}{ext}"
                        save_path = os.path.join(class_path, new_name)
                        img.save(save_path)
                except Exception as e:
                    print(f"  [Lỗi] Không xử lý được ảnh {src_img_path}: {e}")
            print("  -> Đã sinh thêm xong.")

        else:
            print("  -> Số lượng ổn định. Bỏ qua.")

    print("\n--- HOÀN TẤT ---")
    print("Bạn hãy chạy lại file check_data.py để xem kết quả mới.")


if __name__ == "__main__":
    fix_imbalance()