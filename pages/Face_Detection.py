import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def main():
    st.title("Giới thiệu về Phương pháp Face Detection")

    st.header("1. Giới thiệu về Dataset")
    st.write("""
    Dataset được sử dụng trong nghiên cứu này bao gồm:
    - 800 ảnh kích thước 24x24 pixel, trong đó:
        - 400 ảnh là gương mặt người từ nhiều góc độ và điều kiện ánh sáng khác nhau
        - 400 ảnh không phải gương mặt (negative samples)
    - Dataset được chia thành 2 tập:
        - Tập training: 100% số lượng ảnh
        - Tập testing: https://huggingface.co/datasets/Sourabh2/Human_face
    """)

    st.write("Một số mẫu từ tập training:")

    SCALE_FACTOR = 4  # Hệ số phóng đại, ví dụ 24*4 = 96 pixels
    DISPLAY_SIZE = 24 * SCALE_FACTOR  # Kích thước hiển thị cuối cùng

    # Hiển thị ảnh gương mặt với kích thước phóng đại đồng nhất
    st.write("Ảnh gương mặt:")
    face_cols = st.columns(6)
    for i in range(6):
        with face_cols[i]:
            try:
                # Đọc ảnh và resize với hệ số phóng đại
                image = Image.open(f"FaceDetection_process/dataset/face/{i+1}.png")
                image = image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
                st.image(image, 
                        caption=f"Face {i+1}",
                        width=DISPLAY_SIZE,
                        use_column_width=False)
            except FileNotFoundError:
                st.error(f"Không tìm thấy ảnh face/{i+1}.png")

    # Hiển thị ảnh không phải gương mặt với kích thước phóng đại đồng nhất
    st.write("Ảnh không phải gương mặt:")
    nonface_cols = st.columns(6)
    for i in range(6):
        with nonface_cols[i]:
            try:
                # Đọc ảnh và resize với hệ số phóng đại
                image = Image.open(f"FaceDetection_process/dataset/nonface/{i+1}.png")
                image = image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
                st.image(image, 
                        caption=f"Non-face {i+1}",
                        width=DISPLAY_SIZE,
                        use_column_width=False)
            except FileNotFoundError:
                st.error(f"Không tìm thấy ảnh nonface/{i+1}.png")

    st.write("Một số mẫu từ tập test:")
    
    # Khởi tạo list đường dẫn ảnh
    image_paths = [
        "FaceDetection_process/output_face/input_0.jpg",
        "FaceDetection_process/output_face/input_1.jpg",
        "FaceDetection_process/output_face/input_2.jpg",
        "FaceDetection_process/output_face/input_5.jpg",
        "FaceDetection_process/output_face/input_6.jpg",
        "FaceDetection_process/output_face/out_0.png",
        "FaceDetection_process/output_face/out_1.png",
        "FaceDetection_process/output_face/out_2.png",
        "FaceDetection_process/output_face/out_5.png",
        "FaceDetection_process/output_face/out_6.png"
    ]
    
    # Hàng 1
    test_row1 = st.columns(5)
    for i in range(5):
        with test_row1[i]:
            st.image(image_paths[i], caption=f"Test {i}")

    # Hàng 2
    test_row2 = st.columns(5)
    for i in range(5):
        with test_row2[i]:
            st.image(image_paths[i+5], caption=f"Ground Truth {i+5}")

    st.header("2. Giới thiệu Phương pháp")
    st.write("""
    Phương pháp Face Detection sử dụng mô hình Cascade Classifier dựa trên thuật toán Viola-Jones với quy trình như sau:

    - Ảnh Gốc:
        - Đầu vào là ảnh RGB hoặc ảnh xám
        - Được chuyển đổi thành Integral Image để tối ưu tính toán

    - Trích xuất đặc trưng:
        - Sử dụng các hàm Haar cơ bản (edge, line, center-surround)
        - Tính toán đặc trưng thông qua Integral Image
        - Tạo ra số lượng lớn đặc trưng sau khi trích xuất

    - Huấn luyện phân loại:
        - Sử dụng AdaBoost kết hợp kNN để chọn đặc trưng tốt nhất
        - Xây dựng bộ phân loại mạnh từ nhiều bộ phân loại yếu
        - Tối ưu hóa ngưỡng quyết định

    - Cấu trúc phân tầng quyết định:
        - Quét cửa sổ trượt trên ảnh
        - Áp dụng cascade classifier tại mỗi vị trí
        - Đưa ra quyết định có/không phải khuôn mặt
    """)

    st.image("FaceDetection_process/metric_process/Ảnh Gốc.jpg", caption="Minh họa quy trình Face Detection")

    st.header("3. Tham số huấn luyện")
    st.write("""
    Các tham số chính trong quá trình huấn luyện mô hình kết hợp:

    1. **Tham số dữ liệu**:
        - `numPos`: 320 mẫu (80% của 400 ảnh khuôn mặt)
        - `numNeg`: 320 mẫu (80% của 400 ảnh không phải khuôn mặt)
        - `numStages`: 6 stages
        - `w`, `h`: 24x24 pixels

    2. **Tham số Cascade**:
        - `featureType`: `BASIC` (sử dụng các Haar features cơ bản)
        - `bt`: `GAB` (Gentle AdaBoost)
        - `minHitRate`: 0.995 (tỷ lệ hit tối thiểu cho mỗi stage)
        - `maxFalseAlarmRate`: 0.5 (tỷ lệ false alarm tối đa cho mỗi stage)
        - `numFeatures`: 16 features được sử dụng

    3. **Tham số kNN**:
        - `k`: chạy từ 1 đến 31 (step = 2)
        - `distance_metric`: `euclidean` (khoảng cách Euclidean)
        - `weights`: `distance` (trọng số theo khoảng cách)

    **Kết quả cấu hình cascade:**
    - Kích thước cửa sổ huấn luyện: 24x24 pixels
    - Số stages trong cascade: 6 stages
    - Số features được sử dụng: 16 features
    - Cấu trúc cascade: Phân tầng với 6 stages xếp tầng, mỗi stage sử dụng một tập con của 16 features
        + Stage 0: 3 features
        + Stage 1: 3 features  
        + Stage 2: 3 features
        + Stage 3: 3 features
        + Stage 4: 3 features
        + Stage 5: 1 feature
    """)

    st.header("4. Phương pháp Đánh giá")
    st.write("""
    **IoU (Intersection over Union) là gì?**
    
    IoU là cách đo độ chính xác của việc phát hiện khuôn mặt bằng cách so sánh:
    - Vùng chồng lấp giữa khung dự đoán và khung thực tế
    - Càng chồng lấp nhiều càng chính xác

    **Cách tính IoU:**
    - Tính phần giao (vùng chồng lấp)
    - Tính phần hợp (tổng diện tích trừ phần chồng lấp)
    - IoU = Phần giao / Phần hợp

    **Đánh giá kết quả:**
    - IoU = 1.0: Dự đoán hoàn hảo (2 khung trùng nhau)
    - IoU > 0.5: Dự đoán tốt
    - IoU = 0: Dự đoán sai hoàn toàn (không chồng lấp)
    """)

    st.image("FaceDetection_process/metric_process/OIP.jpg", caption="Minh họa cách tính IoU giữa Predicted Box và Ground Truth Box")

    st.header("5. Kết quả Huấn luyện và Test")

    # Phần 1: Kết quả huấn luyện
    st.subheader("5.1 Kết quả Huấn luyện")
    
    # Hàng 1: Biểu đồ
    st.image("FaceDetection_process/FD.png", caption="Biểu đồ hiệu suất KNN với Haar Features")

    # Hàng 2: Phân tích chi tiết
    st.write("""
    **Kết quả phân tích:**
    - Giá trị k tốt nhất: k=13
    - IoU tương ứng: 0.2158 (≈ 0.216)
    """)

    # Phần 2: Kết quả test
    st.subheader("5.2 Kết quả Test trên Tập Test")
    
    # Hàng 1: Ảnh test gốc
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(5)

    with row1_col1:
        st.image("FaceDetection_process/output_face/input_0.jpg", caption="Ảnh test 1", use_column_width=True)

    with row1_col2:
        st.image("FaceDetection_process/output_face/input_1.jpg", caption="Ảnh test 2", use_column_width=True)

    with row1_col3:
        st.image("FaceDetection_process/output_face/input_2.jpg", caption="Ảnh test 3", use_column_width=True)

    with row1_col4:
        st.image("FaceDetection_process/output_face/input_3.jpg", caption="Ảnh test 4", use_column_width=True)
        
    with row1_col5:
        st.image("FaceDetection_process/output_face/input_4.jpg", caption="Ảnh test 5", use_column_width=True)

    # Hàng 2: Kết quả sau khi áp dụng mô hình
    row2_col1, row2_col2, row2_col3, row2_col4, row2_col5 = st.columns(5)

    with row2_col1:
        st.image("FaceDetection_process/output_face/output_0.jpg", caption="Kết quả 1", use_column_width=True)

    with row2_col2:
        st.image("FaceDetection_process/output_face/output_1.jpg", caption="Kết quả 2", use_column_width=True)

    with row2_col3:
        st.image("FaceDetection_process/output_face/output_2.jpg", caption="Kết quả 3", use_column_width=True)

    with row2_col4:
        st.image("FaceDetection_process/output_face/output_3.jpg", caption="Kết quả 4", use_column_width=True)
        
    with row2_col5:
        st.image("FaceDetection_process/output_face/output_4.jpg", caption="Kết quả 5", use_column_width=True)


    st.header("6. Ứng dụng Thử nghiệm")
    st.write("""
    Dưới đây là demo ứng dụng phát hiện khuôn mặt. Bạn có thể tải lên một hình ảnh để thử nghiệm:
    - Hỗ trợ các định dạng: JPG, PNG, JPEG
    - Có thể phát hiện nhiều khuôn mặt trong một ảnh
    - Kết quả hiển thị bounding box màu xanh quanh khuôn mặt
    """)
    
    def apply_face_detection(image):
        # Chuyển đổi sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier('FaceDetection_process/output/cascade.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Vẽ rectangle xung quanh khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return image, len(faces)  # Trả về cả ảnh v số khuôn mặt phát hiện được

    # Phần upload và xử lý ảnh
    uploaded_file = st.file_uploader("Chọn ảnh để nhận diện khuôn mặt", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Hiển thị ảnh gốc và ảnh sau khi xử lý
        col1, col2 = st.columns(2)
        
        # Đọc và hiển thị ảnh gốc
        image = Image.open(uploaded_file)
        with col1:
            st.write("**Ảnh gốc:**")
            st.image(image, use_column_width=True)
        
        # Xử lý ảnh
        img_array = np.array(image)
        processed_img, num_faces = apply_face_detection(img_array)
        
        # Hiển thị ảnh đã xử lý
        with col2:
            st.write("**Ảnh sau khi nhận diện:**")
            st.image(processed_img, use_column_width=True)
            st.write(f"*Số khuôn mặt phát hiện được: {num_faces}*")
        
        # Thêm nút tải ảnh đã xử lý
        processed_img_pil = Image.fromarray(processed_img)
        buf = io.BytesIO()
        processed_img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Tải ảnh đã xử lý",
            data=byte_im,
            file_name="face_detected.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()