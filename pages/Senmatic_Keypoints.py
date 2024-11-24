import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2

def main():
    st.title("Phân tích và Đánh giá Phương pháp Phát hiện Keypoint")

    # Phần 1: Giới thiệu Dataset
    st.header("1. Giới thiệu Dataset")
    st.write("""
    Dataset synthetic được tạo ra để đánh giá hiệu quả của các thuật toán phát hiện góc. Dataset bao gồm các thư mục con sau:

    1. **draw_checkerboard**: Chứa các hình ảnh bàn cờ vua với các ô đen trắng xen kẽ
    2. **draw_cube**: Chứa các hình ảnh khối lập phương 3D
    3. **draw_ellipses**: Chứa các hình ảnh hình elip
    4. **draw_lines**: Chứa các hình ảnh đường thẳng
    5. **draw_multiple_polygons**: Chứa các hình ảnh nhiều đa giác
    6. **draw_polygon**: Chứa các hình ảnh đa giác đơn
    7. **draw_star**: Chứa các hình ảnh hình ngôi sao
    8. **draw_stripes**: Chứa các hình ảnh các dải sọc
    9. **gaussian_noise**: Chứa các hình ảnh nhiễu Gaussian

    Mỗi thư mục con chứa các hình ảnh được tạo ra với các đặc điểm khác nhau, giúp đánh giá khả năng phát hiện góc của thuật toán trong nhiều trường hợp khác nhau.
    """)

    # Danh sách tên file ảnh thực tế
    original_images = [
        "detector_results5/original_images/draw_checkerboard_original.png",
        "detector_results5/original_images/draw_cube_original.png",
        "detector_results5/original_images/draw_lines_original.png",
        "detector_results5/original_images/draw_multiple_polygons_original.png",
        "detector_results5/original_images/draw_polygon_original.png",
        "detector_results5/original_images/draw_star_original.png",
        "detector_results5/original_images/draw_stripes_original.png"
    ]

    gt_images = [
        "detector_results5/draw_checkerboard_result.png",
        "detector_results5/draw_cube_result.png",
        "detector_results5/draw_lines_result.png",
        "detector_results5/draw_multiple_polygons_result.png",
        "detector_results5/draw_polygon_result.png",
        "detector_results5/draw_star_result.png",
        "detector_results5/draw_stripes_result.png"
    ]

    # Style cho container
    st.markdown("""
        <style>
            .stImage > img {
                max-width: 100%;
                height: auto;
            }
            .image-container {
                padding: 10px;
            }
            .center-row {
                display: flex;
                justify-content: center;
                gap: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Hiển thị ảnh gốc
    st.subheader("Ảnh gốc:")
    # Hàng 1: 4 ảnh đầu
    row1_col1 = st.columns(4)
    for i, img_name in enumerate(original_images[:4]):
        with row1_col1[i]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(img_name, caption=f"Hình ảnh gốc {i+1}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Hàng 2: 3 ảnh còn lại, căn giữa
    st.markdown('<div class="center-row">', unsafe_allow_html=True)
    row2_col1 = st.columns([1.5, 2, 2, 2, 1.5])
    for i, img_name in enumerate(original_images[4:7]):
        with row2_col1[i+1]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(img_name, caption=f"Hình ảnh gốc {i+5}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ground Truth với cùng layout
    st.subheader("Ground Truth:")
    # Hàng 1: 4 ảnh đầu
    row1_col2 = st.columns(4)
    for i, gt_name in enumerate(gt_images[:4]):
        with row1_col2[i]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(gt_name, caption=f"Ground Truth {i+1}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Hàng 2: 3 ảnh còn lại, căn giữa
    st.markdown('<div class="center-row">', unsafe_allow_html=True)
    row2_col2 = st.columns([1.5, 2, 2, 2, 1.5])
    for i, gt_name in enumerate(gt_images[4:7]):
        with row2_col2[i+1]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(gt_name, caption=f"Ground Truth {i+5}", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Phần 2: Phương pháp
    st.header("2. Phương pháp")
    
    # 2.1 Giới thiệu phương pháp
    st.subheader("2.1 Giới thiệu phương pháp")
    
    # SIFT
    st.markdown("#### Scale Invariant Feature Transform (SIFT)")
    st.write("""
    SIFT (Scale Invariant Feature Transform) là một thuật toán phát hiện và mô tả đặc trưng cục bộ trong ảnh được phát triển bởi David Lowe vào năm 1999. Đây là một trong những thuật toán quan trọng nhất trong lĩnh vực xử lý ảnh và thị giác máy tính.

    **Ưu điểm chính:**
    - Bất biến với tỷ lệ và phép quay của ảnh
    - Bất biến với thay đổi cường độ sáng
    - Khả năng chống chịu tốt với thay đổi góc nhìn
    - Độ chính xác cao trong việc phát hiện và mô tả đặc trưng

    **Quy trình 4 bước chính:**
    1. **Phát hiện extreme trong không gian tỷ lệ:**
       - Sử dụng Difference of Gaussian (DoG)
       - Tìm các điểm extreme trong không gian tỷ lệ và vị trí

    2. **Định vị keypoint:**
       - Loại bỏ các keypoint có độ tương phản thấp
       - Loại bỏ các điểm nằm trên cạnh
       - Tinh chỉnh vị trí keypoint bằng nội suy

    3. **Gán hướng:**
       - Tính toán magnitude và hướng gradient
       - Tạo histogram hướng gradient
       - Gán hướng chính cho keypoint

    4. **Tạo mô tả đặc trưng:**
       - Tạo vùng 16x16 xung quanh keypoint
       - Chia thành 16 vùng con 4x4
       - Tính histogram hướng gradient cho mỗi vùng
       - Tạo vector đặc trưng 128 chiều
    """)

    # Vị trí để thêm hình ảnh SIFT
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image("ORB_process/SIFT.png", caption="SIFT Keypoints Detection", use_column_width=True)

    # ORB
    st.markdown("#### Oriented FAST and Rotated BRIEF (ORB)")
    st.write("""
    ORB (Oriented FAST and Rotated BRIEF) là một thuật toán được phát triển bởi OpenCV Lab như một giải pháp thay thế miễn phí cho SIFT và SURF. Thuật toán này kết hợp detector FAST và descriptor BRIEF với một số cải tiến.

    **Ưu điểm chính:**
    - Hiệu quả về mặt tính toán và bộ nhớ
    - Tốc độ xử lý nhanh hơn SIFT gấp nhiều lần
    - Không có vấn đề về bản quyền
    - Bất biến với phép quay
    - Phù hợp với ứng dụng thời gian thực

    **Quy trình hoạt động:**
    1. **FAST Keypoint Detector:**
       - Phát hiện góc sử dụng FAST
       - Áp dụng Harris corner measure
       - Sử dụng kim tự tháp để tạo tính bất biến tỷ lệ
       
    2. **Orientation:**
       - Tính toán hướng của patch dựa trên intensity centroid
       - Cải thiện tính bất biến với phép quay

    3. **rBRIEF Descriptor:**
       - Sử dụng BRIEF được cải tiến
       - Xoay pattern theo hướng của keypoint
       - Tạo descriptor nhị phân hiệu quả

    **Ứng dụng:**
    - Nhận dạng đối tượng
    - Theo dõi đối tượng trong video
    - Ghép ảnh panorama
    - SLAM (Simultaneous Localization and Mapping)
    """)

    # Vị trí để thêm hình ảnh ORB
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image("ORB_process/ORB.png", caption="ORB Feature Detection", use_column_width=True)

    # Euclidean Distance
    st.markdown("#### Euclidean Distance")
    st.write("""
    Trong context của đánh giá độ chính xác của việc phát hiện điểm đặc trưng, khoảng cách Euclidean được sử dụng để xác định xem một điểm được phát hiện có thực sự gần với groundtruth hay không.

    **Nguyên lý hoạt động:**
    1. **Vòng tròn Euclidean:**
       - Với mỗi điểm groundtruth, vẽ một vòng tròn với bán kính r
       - Bán kính r là ngưỡng khoảng cách Euclidean được chấp nhận
       - Tạo ra một vùng chấp nhận xung quanh mỗi điểm groundtruth

    2. **Đánh giá độ chính xác:**
       - Một điểm được phát hiện (detected point) được coi là đúng nếu nó nằm trong vòng tròn
       - Công thức tính khoảng cách: d = √[(x₁-x₂)² + (y₁-y₂)²]
       - Nếu d ≤ r: điểm phát hiện được coi là chính xác
       - Nếu d > r: điểm phát hiện được coi là sai

    **Ưu điểm của phương pháp:**
    - Cho phép một độ sai số chấp nhận được
    - Đơn giản và trực quan trong việc đánh giá
    - Phù hợp với nhiều loại dataset khác nhau

    **Các tham số quan trọng:**
    - Bán kính r: quyết định độ nghiêm ngặt của việc đánh giá
    - Càng nhỏ r: đánh giá càng nghiêm ngặt
    - Càng lớn r: cho phép sai số nhiều hơn
    """)

    # Vị trí để thêm hình ảnh Euclidean Distance
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image("results/euclidean_visualization.png", caption="Euclidean Distance Evaluation", use_column_width=True)

    # 2.2 Kết quả thực nghiệm
    st.subheader("2.2 Kết quả thực nghiệm")
    
    # Kết quả ORB
    st.markdown("#### Kết quả ORB")
    orb_files = sorted([f for f in os.listdir("ORB") if f.endswith(('.jpg', '.png', '.jpeg'))])
    cols_orb = st.columns(4)
    for i, image_file in enumerate(orb_files[:20]):  # Giới hạn 16 ảnh
        with cols_orb[i % 4]:
            image_path = os.path.join("ORB", image_file)
            st.image(image_path, caption=f"ORB {i+1}", use_column_width=True)

    # Kết quả SIFT
    st.markdown("#### Kết quả SIFT")
    sift_files = sorted([f for f in os.listdir("SIFT") if f.endswith(('.jpg', '.png', '.jpeg'))])
    cols_sift = st.columns(4)
    for i, image_file in enumerate(sift_files[:20]):  # Giới hạn 16 ảnh
        with cols_sift[i % 4]:
            image_path = os.path.join("SIFT", image_file)
            st.image(image_path, caption=f"SIFT {i+1}", use_column_width=True)

    # Phần 3: Phương pháp đánh giá
    st.header("3. Phương pháp đánh giá")

    # Phần giải thích Precision và Recall
    st.write("""
    #### Precision (Độ chính xác)
    Precision = TP / (TP + FP)
    - TP (True Positive): Số keypoint phát hiện đúng
    - FP (False Positive): Số keypoint phát hiện sai

    #### Recall (Độ phủ)
    Recall = TP / (TP + FN)
    - TP (True Positive): Số keypoint phát hiện đúng
    - FN (False Negative): Số keypoint bỏ sót
    """)

    # Thêm hình minh họa ở cuối
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image("ORB_process/R.png", caption="Minh họa Precision và Recall", use_column_width=True)

    # Phần 4: Kết quả đánh giá
    st.header("4. Kết quả đánh giá")
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image("ORB_process/z6053353024997_2f64f94bcd3e52b73c05bf16c0df3fad.jpg", 
                 caption="Biểu đồ so sánh kết quả đánh giá SIFT và ORB", 
                 use_column_width=True)

    # Phần 5: Thảo luận
    st.header("5. Thảo luận")
    st.write("""
    Dựa trên kết quả thực nghiệm, chúng ta có thể rút ra một số nhận xét sau:

    #### 1. So sánh hiệu năng SIFT và ORB
    - ORB cho hiệu suất tốt hơn SIFT trong hầu hết các trường hợp
    - Precision của ORB đạt cao nhất với draw_checkerboard và draw_polygon (khoảng 0.8)
    - Recall của ORB đặc biệt tốt với draw_star (trên 0.8)
    - SIFT có độ ổn định cao hơn nhưng hiệu suất thấp hơn, với recall dao động từ 0.3 đến 0.5

    #### 2. Ưu và nhược điểm của từng phương pháp
    **SIFT:**
    - Ưu điểm:
      + Độ ổn định cao giữa các loại ảnh khác nhau
      + Ít biến động về hiệu suất
    - Nhược điểm:
      + Hiệu suất thấp hơn ORB
      + Đặc biệt kém hiệu quả với một số loại ảnh như draw_checkerboard

    **ORB:**
    - Ưu điểm:
      + Hiệu suất cao trong hầu hết các trường hợp
      + Đặc biệt hiệu quả với các hình học cơ bản
    - Nhược điểm:
      + Hiệu suất không ổn định giữa các loại ảnh
      + Có sự biến động lớn về recall

    #### 3. Hướng cải thiện và phát triển
    - Kết hợp ưu điểm của cả hai phương pháp
    - Tối ưu hóa tham số cho từng loại ảnh cụ thể
    - Phát triển phương pháp mới dựa trên điểm mạnh của ORB
    - Nghiên cứu thêm về khả năng xử lý với các loại ảnh phức tạp
    - Cải thiện độ ổn định của ORB và hiệu suất của SIFT
    """)

if __name__ == '__main__':
    main()