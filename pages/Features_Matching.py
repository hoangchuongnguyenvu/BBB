import streamlit as st
import os
from glob import glob
import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    st.set_page_config(page_title="SuperPoint Analysis", page_icon="📊")
    
    # Tiêu đề chính
    st.title("Phân tích và đánh giá SuperPoint")
    
    # 1. Phần giới thiệu dataset
    st.header("1. Giới thiệu Dataset")
    
    # Container cho 8 hình ảnh phía trên
    st.write("#### Hình ảnh minh họa các keypoints được phát hiện")
    
    # Tạo 7 cột cho mỗi hàng
    cols_top = st.columns(7)
    
    # Danh sách đường dẫn đến các hình ảnh của bạn
    image_paths = [
        "detector_results5/original_images/draw_checkerboard_original.png",
        "detector_results5/original_images/draw_cube_original.png",
        "detector_results5/original_images/draw_polygon_original.png",
        "detector_results5/original_images/draw_lines_original.png",
        "detector_results5/original_images/draw_multiple_polygons_original.png",
        "detector_results5/original_images/draw_star_original.png",
        "detector_results5/original_images/draw_stripes_original.png",
        # Hàng 2
        "detector_results5/draw_checkerboard_result.png",
        "detector_results5/draw_cube_result.png",
        "detector_results5/draw_polygon_result.png",
        "detector_results5/draw_lines_result.png",
        "detector_results5/draw_multiple_polygons_result.png",
        "detector_results5/draw_star_result.png",
        "detector_results5/draw_stripes_result.png",
    ]
    
    # Hiển thị hàng đầu tiên (7 ảnh)
    for i, col in enumerate(cols_top):
        with col:
            st.image(
                image_paths[i],
                caption=f"Ảnh gốc {i + 1}",
                use_column_width=True
            )
    
    # Tạo 7 cột cho hàng thứ hai
    cols_bottom = st.columns(7)
    
    # Hiển thị hàng thứ hai (7 ảnh)
    for i, col in enumerate(cols_bottom):
        with col:
            st.image(
                image_paths[i + 7],  # Lấy 7 ảnh tiếp theo
                caption=f"Keypoints Groundtruth {i + 8}",
                use_column_width=True
            )
    
    # Nội dung giới thiệu dataset
    st.write("""
    ### Tổng quan về Synthetic Dataset
    
    Dataset tổng hợp được tạo ra để huấn luyện và đánh giá mô hình SuperPoint, bao gồm nhiều loại hình học cơ bản khác nhau:

    1. **Cấu trúc Dataset:**
    Dataset bao gồm các thư mục con sau: draw_checkerboard, draw_cube, draw_ellipses, draw_lines, draw_multiple_polygons, draw_polygon, draw_star, draw_stripes, gaussian_noise

    Mỗi thư mục con đều chứa:
    - Thư mục `images/`: Lưu trữ các ảnh synthetic
    - Thư mục `points/`: Chứa các file .npy tương ứng lưu tọa độ groundtruth keypoints của mỗi ảnh
    
    2. **Đặc điểm của Dataset:**
    - Mỗi thư mục con đại diện cho một loại hình học khác nhau (đường thẳng, đa giác, hình sao,...)
    - Tất cả ảnh được tạo tự động (synthetic) với các keypoints được xác định chính xác
    - Mỗi ảnh đều có một file .npy tương ứng chứa thông tin về vị trí các keypoints

    3. **Format dữ liệu:**
    - Ảnh được lưu dưới dạng grayscale với kích thước 240x320 pixels
    - File .npy chứa mảng numpy 2 chiều với shape (N, 2), trong đó N là số lượng keypoints
    - Mỗi keypoint được biểu diễn bởi tọa độ (x, y) trong ảnh
    """)
    

    # 2. Phần phương pháp
    st.header("2. Phương pháp")
    st.write("""
    ### Giới thiệu về SuperPoint
    SuperPoint là một mô hình deep learning được giới thiệu vào năm 2018 trong bài báo "SuperPoint: Self-Supervised Interest Point Detection and Description" bởi DeTone, Malisiewicz và Rabinovich tại Magic Leap, Inc. Bài báo được công bố tại hội nghị CVPR Workshop 2018.

    ### Đặc điểm nổi bật
    - Là một trong những mô hình đầu tiên áp dụng deep learning vào bài toán phát hiện và mô tả đặc trưng
    - Sử dụng phương pháp học tự giám sát (self-supervised learning)
    - Có khả năng hoạt động real-time
    """)

    # Hiển thị hình ảnh kiến trúc mạng
    st.write("### Kiến trúc tổng quan của SuperPoint")
    st.image("keke.jpg", caption="Kiến trúc mạng SuperPoint", use_column_width=True)

    st.write("""
    ### Kiến trúc mạng
    SuperPoint được thiết kế với kiến trúc đa nhiệm, bao gồm:

    1. **Shared Encoder:**
    - Sử dụng kiến trúc VGG-style
    - 8 lớp tích chập (convolutional layers)
    - Giảm độ phân giải ảnh xuống 1/8
    - Nhận đầu vào là ảnh grayscale kích thước H×W×1
    
    2. **Interest Point Decoder (Detector Head):**
    - Phát hiện các điểm đặc trưng (keypoints)
    - Tạo ra heatmap với kích thước H/8 × W/8 × 65
    - Sử dụng lớp Softmax để tính xác suất điểm đặc trưng
    - Reshape để tạo bản đồ điểm đặc trưng cuối cùng
    
    3. **Descriptor Decoder (Descriptor Head):**
    - Tạo vector đặc trưng cho mỗi keypoint
    - Kích thước descriptor: 256 chiều (D=256)
    - Sử dụng phép nội suy Bi-Cubic
    - Chuẩn hóa L2 để tạo ra các descriptor có độ dài cố định

    ### Quy trình xử lý
    1. Ảnh đầu vào được đưa qua Shared Encoder để trích xuất đặc trưng
    2. Đặc trưng được chia thành 2 nhánh xử lý song song:
       - Nhánh Interest Point Decoder: phát hiện vị trí các điểm đặc trưng
       - Nhánh Descriptor Decoder: tạo ra các vector mô tả cho mỗi điểm
    """)
    
    # Định nghĩa style cho container
    container_style = """
        <div style='display: flex; flex-direction: column; height: 500px; align-items: center;'>
            <h4 style='text-align: center; height: 50px; margin: 10px 0;'>
                {}
            </h4>
            <div style='flex: 1; display: flex; align-items: center; width: 100%; padding: 10px;'>
                <img src='{}' style='width: 100%; max-height: 350px; object-fit: contain;'>
            </div>
            <p style='text-align: center; height: 30px; margin: 10px 0;'>
                {}
            </p>
        </div>
    """
    # Hiển thị hình ảnh ở giữa
    st.image("OIP.jpg", 
             caption="Quá trình trích xuất đặc trưng", 
             use_column_width=True)
    
    # 3. Phần phương pháp đánh giá
    st.header("3. Phương pháp đánh giá")
    st.write("""
    ### Quy trình đánh giá
    Để đánh giá hiệu suất của SuperPoint và so sánh với các phương pháp truyền thống như SIFT và ORB trên tập dữ liệu Synthetic Shapes Dataset, chúng tôi thực hiện các bước sau:

    1. **Trích xuất đặc trưng:**
    - Trích xuất vector đặc trưng sử dụng SIFT, ORB và SuperPoint
    - Thực hiện trên các keypoint ground truth ở các góc quay khác nhau của ảnh
    
    2. **So khớp keypoint:**
    - Sử dụng phương pháp Brute-Force Matching để so khớp các vector đặc trưng
    - So sánh giữa ảnh gốc và các ảnh đã quay
    
    3. **Đánh giá kết quả:**
    - Tính toán phần trăm các keypoint được so khớp chính xác
    - So sánh hiệu suất giữa các phương pháp ở mỗi góc quay
    - Phân tích độ ổn định của các phương pháp dưới ảnh hưởng của phép quay
    """)

    # Hiển thị hình ảnh minh họa quy trình đánh giá
    st.image("34e1ed465aa5420d82438c0e6ad330a2.png", 
             caption="Quá trình so khớp các đặc trưng", 
             use_column_width=True)
    
    # 4. Phần kết quả thí nghiệm
    st.header("4. Kết quả thí nghiệm")
    st.write("### Kết quả so sánh hiệu suất")

    # Hiển thị hình ảnh từ thư mục của project
    st.image("matching.png", caption='Biểu đồ so sánh hiệu suất')

    st.write("""
    ### Đánh giá chi tiết
    
    **1. SuperPoint:**
    - Có hiệu suất tốt nhất ở các góc quay nhỏ (0° đến 30°)
    - Duy trì độ chính xác cao nhất trong khoảng 10° đến 30° (khoảng 0.93 đến 0.73)
    - Tuy nhiên, hiệu suất giảm mạnh sau góc 30° và duy trì ở mức thấp
    - Phù hợp cho các ứng dụng có góc quay nhỏ cần độ chính xác cao

    **2. SIFT:**
    - Có hiệu suất thấp nhất trong 3 phương pháp
    - Độ chính xác giảm đều và nhanh theo góc quay
    - Khá ổn định ở góc nhỏ nhưng kém hiệu quả khi góc quay tăng
    - Ít phù hợp cho các ứng dụng có góc quay lớn

    **3. ORB:**
    - Có hiệu suất trung bình ở góc nhỏ, nhưng duy trì độ ổn định tốt hơn khi góc tăng
    - Từ góc 40° trở đi, ORB cho hiệu suất tốt hơn cả SuperPoint và SIFT
    - Có độ ổn định tốt nhất trong 3 phương pháp khi góc quay lớn
    - Phù hợp cho các ứng dụng cần xử lý ảnh ở nhiều góc quay khác nhau

    **Kết luận chung:**
    - Mỗi phương pháp có ưu điểm riêng ở các khoảng góc quay khác nhau
    - SuperPoint tốt nhất cho góc nhỏ, ORB phù hợp với góc lớn
    - Cả 3 phương pháp đều giảm hiệu suất đáng kể khi góc quay tăng
    - Việc lựa chọn phương pháp phụ thuộc vào yêu cầu cụ thể của ứng dụng
    """)

    # Tạo slider để chọn góc
    angle = st.slider("Chọn góc quay", 0, 60, 0, 10)

    # CSS cho tooltip
    st.markdown("""
        <style>
        .tooltip-container {
            position: relative;
            display: inline-block;
        }
        .tooltip-container .tooltip-content {
            visibility: hidden;
            background-color: #f0f2f6;
            color: black;
            text-align: left;
            padding: 10px;
            border-radius: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            width: 200px;
        }
        .tooltip-container:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    # Tạo grid 3x3 để hiển thị ảnh
    cols = st.columns(3)
    
    # Danh sách các thư mục con
    subdirs = [
        "draw_checkerboard",
        "draw_cube",
        "draw_lines",
        "draw_multi_polygons",
        "draw_polygons",
        "draw_star",
        "draw_stripes"
    ]
    
    # Duyệt qua từng thư mục con
    for idx, subdir in enumerate(subdirs):
        pattern = f"KQ/{subdir}/{angle}_*_*_*.png"
        matching_images = glob(pattern)
        
        if matching_images:
            img_path = matching_images[0]
            x1, x2, x3, x4 = os.path.basename(img_path).split('.')[0].split('_')
            
            with cols[idx % 3]:
                # Tạo container với tooltip
                st.markdown(f"""
                    <div class="tooltip-container">
                        <img src="data:image/png;base64,{get_image_base64(img_path)}" style="width:100%">
                        <div class="tooltip-content">
                            <b>Thông số:</b><br>
                            • Tổng matches: {x4}<br>
                            • Good matches: {x2}<br>
                            • Matches detect: {x3}
                        </div>
                    </div>
                    <p style="text-align:center">Kết quả từ {subdir}</p>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
