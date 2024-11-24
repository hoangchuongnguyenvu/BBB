import streamlit as st
import cv2
import numpy as np
from PIL import Image

def main():
    st.title("Giới thiệu về Phương pháp Watershed")

    st.header("1. Giới thiệu về Dataset")
    st.write("Dataset được sử dụng trong nghiên cứu này bao gồm các hình ảnh biển số xe từ nhiều góc độ và điều kiện ánh sáng khác nhau.")

    st.subheader("Dataset")
    
    # Tạo container cho tiêu đề
    title_col1, title_col2 = st.columns(2)
    with title_col1:
        st.markdown("#### 1.1. Tập train")
    with title_col2:
        st.markdown("#### 1.2. Tập test")
    
    # Tạo 4 cột cho hàng ảnh gốc
    img_cols = st.columns(4)
    
    # Hiển thị ảnh gốc với width cố định
    with img_cols[0]:
        st.image("Watershed_Streamlit/train_test/1xemay356.jpg", 
                caption="Ảnh 1 trong tập train",
                width=150)
    with img_cols[1]:
        st.image("Watershed_Streamlit/train_test/ndata217_train.jpg", 
                caption="Ảnh 2 trong tập train",
                width=150)
    with img_cols[2]:
        st.image("Watershed_Streamlit/train_test/ndata165_train.jpg", 
                caption="Ảnh 1 trong tập test",
                width=150)
    with img_cols[3]:
        st.image("Watershed_Streamlit/train_test/ndata227.jpg", 
                caption="Ảnh 2 trong tập test",
                width=150)
    
    # Thêm khoảng trống giữa hai hàng
    st.write("")
    
    # Tạo 4 cột cho hàng ground truth
    gt_cols = st.columns(4)
    
    # Hiển thị ground truth với width cố định
    with gt_cols[0]:
        st.image("Watershed_Streamlit/train_test/1xemay356.png", 
                caption="Ground truth của ảnh 1 trong tập train",
                width=150)
    with gt_cols[1]:
        st.image("Watershed_Streamlit/train_test/ndata217_train_gt.png", 
                caption="Ground truth của ảnh 2 trong tập train",
                width=150)
    with gt_cols[2]:
        st.image("Watershed_Streamlit/train_test/ndata165_train_gt.png", 
                caption="Ground truth của ảnh 1 trong tập test",
                width=150)
    with gt_cols[3]:
        st.image("Watershed_Streamlit/train_test/ndata227.png", 
                caption="Ground truth của ảnh 2 trong tập test",
                width=150)

    st.header("2. Giới thiệu Phương pháp")
    
    # Hiển thị hình ảnh minh họa trước
    
    
    # Phần mô tả phương pháp
    st.write("""
### 2.1. Phương pháp Watershed
Phương pháp Watershed là kỹ thuật phân đoạn hình ảnh dựa trên nguyên lý thủy văn, trong đó hình ảnh được xem như một bề mặt địa hình với các điểm có cường độ pixel thấp được xem như thung lũng và các điểm có cường độ pixel cao được xem như đỉnh núi. Trong bài toán nhận dạng biển số xe, phương pháp này được sử dụng để tách biệt các ký tự trên biển số thông qua các bước sau:

- Tiền xử lý dữ liệu:
    - Chuyển đổi ảnh gốc (Skewed Image) sang ảnh xám (Grayscale Image)
    - Mục đích: Giảm độ phức tạp của ảnh và chuẩn bị cho các bước xử lý tiếp theo

- Tiền phân đoạn:
    - Thực hiện phân ngưỡng nhị phân (Binarization) để tách biệt đối tượng và nền
    - Kết quả: Ảnh nhị phân với biển số xe được tách biệt rõ ràng

- Loại bỏ nhiễu:
    - Áp dụng các phép toán hình thái học (Morphological Opening)
    - Mục đích: Loại bỏ các nhiễu nhỏ và làm mịn biên của đối tượng

- Tính toán khoảng cách:
    - Áp dụng phép biến đổi khoảng cách (Distance Transform)
    - Tính toán khoảng cách từ mỗi điểm ảnh đến điểm gần nhất

- Xác định vùng đối tượng:
    - Xác định Sure Background (vùng chắc chắn là nền)
    - Xác định Sure Foreground (vùng chắc chắn là đối tượng)
    - Xác định Unknown Region (vùng không chắc chắn)
    - Kết hợp các vùng để tạo markers cho thuật toán Watershed
    """)
    st.image("Watershed_Streamlit/metric_process/process.jpg", caption="Minh họa quá trình Watershed trên biển số xe")
    st.write("""
### 2.2. Kỹ thuật Phân đoạn Ký tự
Sau khi áp dụng Watershed, chúng tôi sử dụng kỹ thuật phân đoạn ký tự dựa trên đặc điểm hình học của các ký tự trên biển số xe:

1. Tìm Contours:
    - Sử dụng cv2.findContours để tìm các đường viền của các vùng đã được phân đoạn
    - Mỗi contour tiềm năng đại diện cho một ký tự riêng biệt

2. Lọc Ký tự dựa trên Đặc điểm Hình học:
    - Tỷ lệ khung hình (Aspect Ratio):
        * Tính tỷ lệ chiều rộng/chiều cao (w/h) của mỗi contour
        * Chỉ giữ lại các contour có tỷ lệ trong khoảng 0.1 < w/h < 0.9
        * Điều kiện này dựa trên đặc điểm của ký tự trên biển số: thường cao hơn rộng
    
    - Diện tích (Area):
        * Tính diện tích trung bình của tất cả contours
        * Loại bỏ các contour có diện tích nhỏ hơn 20% diện tích trung bình
        * Giúp loại bỏ nhiễu và các vùng không phải ký tự

    """)

    st.header("3. Tham số huấn luyện")
    st.write("""
    Các tham số chính trong thuật toán Watershed:

1. `threshold_value`: Giá trị ngưỡng cho quá trình phân đoạn ban đầu
2. `opening_iterations`: Số lần lặp mở để loại bỏ nhiễu
3. `dilation_iterations`: Số lần lặp giãn nở để xác định vùng nền chắc chắn
    """)

    st.markdown("""
    ### Tham số quan trọng nhất:
    > **`dist_threshold`** (Ngưỡng khoảng cách)
    > - Phạm vi: 0.0 → 0.4
    > - Bước nhảy: 0.01
    > - Công thức: `threshold = dist_threshold * dist_transform.max()`
    >
    > **kernel_size** (Kích thước kernel)
    > - Kích thước của kernel được sử dụng trong các phép toán hình thái học
    > - Các giá trị kernel được thử nghiệm:
    >   * `3x3`
    >   * `5x5`
    >   * `7x7`
    >   * `9x9`
    >   * `11x11`
    """)

    st.write("Các tham số này có thể được điều chỉnh để tối ưu hóa kết quả phân đoạn cho các loại hình ảnh khác nhau.")

    st.header("4. Phương pháp Đánh giá")
    st.write("""
    Để đánh giá hiệu quả của thuật toán Watershed trong việc phân đoạn biển số xe, nghiên cứu sử dụng metric Intersection over Union (IoU). Đây là một trong những phương pháp đánh giá phổ biến nhất trong các bài toán phân đoạn ảnh.
    IoU được tính bằng tỷ số giữa diện tích phần giao (Intersection) và diện tích phần hợp (Union) của hai vùng:
    - Vùng được phân đoạn bởi thuật toán (Predicted region)
    - Vùng ground truth (Ground Truth)
    
    Đặc điểm:
    - Giá trị ∈ [0, 1]
    - IoU = 1: phân đoạn hoàn hảo
    - IoU > 0.5: kết quả chấp nhận được
    """)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("Watershed_Streamlit/metric_process/OIP.jpg", caption="Minh họa cách tính IoU")

    st.header("5. Kết quả Huấn luyện và Test")

    # Phần 1: Kết quả huấn luyện
    st.subheader("5.1 Kết quả Huấn luyện")
    
    # Hiển thị biểu đồ đường
    st.image("Watershed_Streamlit/metric_process/iou_curves.png", caption="Biểu đồ kết quả huấn luyện")
    
    st.write("""
    Từ biểu đồ trên, chúng ta có thể thấy kết quả tốt nhất đạt được với:
    - Kernel size: 3x3
    - Distance threshold: 0.02
    
    Với bộ tham số này, thuật toán cho kết quả phân đoạn tốt nhất trên tập huấn luyện. Kernel size nhỏ (3x3) giúp bảo toàn các chi tiết nhỏ của ký tự, trong khi distance threshold thấp (0.02) giúp phát hiện chính xác các vùng foreground của ký tự.
    """)
    
    # Thêm phần điều chỉnh tham số
    st.subheader("Điều chỉnh tham số Watershed")

    # Hiển thị thanh trượt tham số
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        distance_threshold = st.slider(
            "Distance Threshold",
            min_value=0.0,
            max_value=0.4,
            value=0.2,
            step=0.01
        )
    with col_params2:
        kernel_size = st.slider(
            "Kernel Size",
            min_value=3,
            max_value=11,
            value=5,
            step=2
        )

    # Đọc ảnh training
    @st.cache_data
    def load_images():
        train_img1 = cv2.imread('Watershed_Streamlit/train_test/1xemay356.jpg')
        train_gt1 = cv2.imread('Watershed_Streamlit/train_test/1xemay356.png', 0)
        train_img2 = cv2.imread('Watershed_Streamlit/train_test/ndata217_train.jpg')
        train_gt2 = cv2.imread('Watershed_Streamlit/train_test/ndata217_train_gt.png', 0)
        return train_img1, train_gt1, train_img2, train_gt2

    train_img1, train_gt1, train_img2, train_gt2 = load_images()

    # Tính toán trước cho cả hai ảnh
    # Image 1
    resized_img1 = resize_image(train_img1)
    result1 = apply_watershed(resized_img1, distance_threshold, kernel_size)
    resized_gt1 = resize_image(train_gt1)
    iou1 = calculate_iou(result1, resized_gt1)

    # Image 2
    resized_img2 = resize_image(train_img2)
    result2 = apply_watershed(resized_img2, distance_threshold, kernel_size)
    resized_gt2 = resize_image(train_gt2)
    iou2 = calculate_iou(result2, resized_gt2)

    # Training Image 1
    st.markdown("### Training Image 1")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    with col1:
        st.image(cv2.cvtColor(resized_img1, cv2.COLOR_BGR2RGB), caption="Original Image")
    with col2:
        st.image(result1, caption="Prediction")
    with col3:
        st.image(resized_gt1, caption="Ground Truth")
    with col4:
        st.markdown("**Parameters:**")
        st.markdown(f"- Distance Threshold: {distance_threshold:.3f}")
        st.markdown(f"- Kernel Size: {kernel_size}x{kernel_size}")
        st.markdown(f"- IoU: {iou1:.4f}")

    # Hiển thị Average IoU ở giữa
    st.markdown(f"<h3 style='text-align: center;'>Average IoU: {(iou1 + iou2) / 2:.4f}</h3>", unsafe_allow_html=True)

    # Training Image 2
    st.markdown("### Training Image 2")
    col5, col6, col7, col8 = st.columns([2, 2, 2, 2])
    
    with col5:
        st.image(cv2.cvtColor(resized_img2, cv2.COLOR_BGR2RGB), caption="Original Image")
    with col6:
        st.image(result2, caption="Prediction")
    with col7:
        st.image(resized_gt2, caption="Ground Truth")
    with col8:
        st.markdown("**Parameters:**")
        st.markdown(f"- Distance Threshold: {distance_threshold:.3f}")
        st.markdown(f"- Kernel Size: {kernel_size}x{kernel_size}")
        st.markdown(f"- IoU: {iou2:.4f}")

    # Phần 2: Kết quả test
    st.subheader("5.2 Kết quả Test trên Tập Test")
    
    # Test Image 1
    col_test1, col_test2 = st.columns([4, 1])
    with col_test1:
        st.image("Watershed_Streamlit/metric_process/test1.png", caption="Ảnh test 1")
    with col_test2:
        st.markdown("**Parameters:**")
        st.markdown(f"- Kernel: 3x3")
        st.markdown(f"- Threshold: {distance_threshold:.3f}")
        st.markdown(f"- IoU: 0.7316")

    # Test Image 2
    col_test3, col_test4 = st.columns([4, 1])
    with col_test3:
        st.image("Watershed_Streamlit/metric_process/test2.png", caption="Ảnh test 2")
    with col_test4:
        st.markdown("**Parameters:**")
        st.markdown(f"- Kernel: 3x3")
        st.markdown(f"- Threshold: {distance_threshold:.3f}")
        st.markdown(f"- IoU: 0.8608")

def apply_watershed(image, distance_threshold, kernel_size):
    # Copy toàn bộ hàm apply_watershed từ kaks.py
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    _, sure_fg = cv2.threshold(dist_transform, distance_threshold * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    
    result = np.zeros_like(gray)
    result[markers > 1] = 255
    
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_result = np.zeros_like(gray)
    
    if len(contours) > 0:
        avg_area = np.mean([cv2.contourArea(cnt) for cnt in contours])
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            area = cv2.contourArea(cnt)
            
            if 0.1 < aspect_ratio < 0.9 and area > avg_area * 0.2:
                cv2.drawContours(final_result, [cnt], -1, 255, -1)
    
    return final_result

def calculate_iou(pred, gt):
    # Copy toàn bộ hàm calculate_iou từ kaks.py
    pred = pred > 0
    gt = gt > 0
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 0
    
    return intersection / union

# Thêm hàm resize_image
def resize_image(image, target_size=(400, 300)):
    """Resize image while maintaining aspect ratio"""
    if len(image.shape) == 2:  # Grayscale image
        h, w = image.shape
    else:  # Color image
        h, w = image.shape[:2]
    
    # Tính tỷ lệ resize
    target_w, target_h = target_size
    scale = min(target_w/w, target_h/h)
    
    # Tính kích thước mới
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize ảnh
    if len(image.shape) == 2:  # Grayscale image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:  # Color image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized

if __name__ == "__main__":
    main()