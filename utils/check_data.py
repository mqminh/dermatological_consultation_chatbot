import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CẤU HÌNH
# ==========================================
DATA_DIR = '../dataset'  # Đổi tên folder nếu cần
OUTPUT_IMG = 'data_distribution.png'  # Tên file biểu đồ sẽ lưu


def analyze_dataset(data_dir):
    if not os.path.exists(data_dir):
        print(f"Lỗi: Không tìm thấy thư mục '{data_dir}'")
        return

    print(f"--- ĐANG KIỂM TRA DỮ LIỆU TẠI: {data_dir} ---\n")

    class_names = []
    counts = []

    # Duyệt qua các folder con
    try:
        subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    except Exception as e:
        print(f"Lỗi đọc thư mục: {e}")
        return

    if not subfolders:
        print("Không tìm thấy class nào (folder con) trong dataset.")
        return

    # Đếm số lượng ảnh trong từng class
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        # Chỉ đếm các file ảnh phổ biến
        num_files = len([name for name in os.listdir(folder)
                         if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

        class_names.append(folder_name)
        counts.append(num_files)
        print(f" -> Class '{folder_name:<20}': {num_files} ảnh")

    # Tính toán thống kê
    total_images = sum(counts)
    if total_images == 0:
        print("\nDataset rỗng!")
        return

    max_count = max(counts)
    min_count = min(counts)
    avg_count = np.mean(counts)

    print("\n" + "=" * 40)
    print("KẾT QUẢ THỐNG KÊ (Dùng cho báo cáo)")
    print("=" * 40)
    print(f"Tổng số ảnh        : {total_images}")
    print(f"Số lượng classes   : {len(class_names)}")
    print(f"Class nhiều ảnh nhất: {class_names[counts.index(max_count)]} ({max_count} ảnh)")
    print(f"Class ít ảnh nhất   : {class_names[counts.index(min_count)]} ({min_count} ảnh)")
    print(f"Trung bình         : {avg_count:.1f} ảnh/class")

    # Đánh giá độ mất cân bằng
    ratio = max_count / min_count
    print(f"\nĐộ lệch (Max/Min)  : {ratio:.2f} lần")

    if ratio < 1.5:
        print("=> ĐÁNH GIÁ: Dữ liệu cân bằng tốt (Balanced).")
    elif ratio < 3.0:
        print("=> ĐÁNH GIÁ: Dữ liệu mất cân bằng nhẹ (Slightly Imbalanced).")
        print("   Khuyên dùng: Data Augmentation cơ bản.")
    else:
        print("=> ĐÁNH GIÁ: Dữ liệu mất cân bằng NGHIÊM TRỌNG (Highly Imbalanced).")
        print("   Khuyên dùng: Bắt buộc dùng Class Weights hoặc Oversampling.")

    # ==========================================
    # VẼ BIỂU ĐỒ (VISUALIZATION)
    # ==========================================
    plt.figure(figsize=(12, 6))

    # Biểu đồ cột
    plt.subplot(1, 2, 1)
    bars = plt.bar(class_names, counts, color='skyblue', edgecolor='black')
    plt.title('Số lượng ảnh từng bệnh', fontsize=14)
    plt.xlabel('Tên bệnh')
    plt.ylabel('Số lượng')
    plt.xticks(rotation=45, ha='right')

    # Hiển thị số trên đầu cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, int(yval), ha='center', va='bottom')

    # Biểu đồ tròn
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
    plt.title('Tỷ lệ phân bố dữ liệu', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"\n[INFO] Đã lưu biểu đồ phân tích vào file: {OUTPUT_IMG}")
    print("Hãy mở file ảnh này để dán vào báo cáo.")
    plt.show()


if __name__ == "__main__":
    analyze_dataset(DATA_DIR)