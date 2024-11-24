import streamlit as st

def main():
    st.title("SORT - Simple Online and Realtime Tracking")
    
    # 1. Giới thiệu
    st.header("1. Giới thiệu")
    
    # 1.1. Tổng quan
    st.subheader("1.1. Tổng quan") 
    st.markdown("""
    - **SORT** (Simple Online and Realtime Tracking) là một thuật toán theo dõi nhiều đối tượng (Multi-Object Tracking - MOT) 
    được phát triển với mục tiêu đơn giản hóa quá trình tracking nhưng vẫn đảm bảo hiệu suất cao cho ứng dụng thời gian thực.
    SORT được thiết kế để giải quyết bài toán theo dõi đa đối tượng một cách hiệu quả và đơn giản, phù hợp cho các ứng dụng 
    thực tế đòi hỏi xử lý thời gian thực như giám sát an ninh, phân tích giao thông hay phân tích hành vi.

    - **Hoàn cảnh ra đời:**
    Trong bối cảnh các thuật toán tracking đang ngày càng trở nên phức tạp, cộng đồng Computer Vision cần một giải pháp đơn giản 
    nhưng vẫn đảm bảo hiệu quả cho các ứng dụng thời gian thực.

        - *Thời điểm công bố:* Năm 2016

        - *Nơi công bố:* IEEE International Conference on Image Processing (ICIP)

        - *Tác giả:* Alex Bewley và các cộng sự

        - *Tham khảo:* [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)

    - **Đặc điểm nổi bật**:
        - Tốc độ xử lý cực nhanh: >260 FPS trên CPU đơn nhân, vượt trội so với các phương pháp cùng thời
        - Kiến trúc đơn giản: kết hợp Kalman Filter và Hungarian Algorithm một cách hiệu quả
        - Độ chính xác (MOTA) cạnh tranh với các phương pháp phức tạp hơn nhiều
        - Không yêu cầu GPU hay deep learning cho quá trình tracking
        - Dễ dàng tích hợp với bất kỳ object detector nào
        - Mã nguồn mở và được cộng đồng phát triển mạnh mẽ
    """)

    # Hiển thị bảng so sánh
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('SORT_process/bsbsbs.png', caption='Bảng so sánh hiệu suất của SORT với các phương pháp khác')

    # Thêm phần nhận xét về hiệu suất
    st.markdown("""
    - **Phân tích hiệu suất của SORT**:
        - **Độ chính xác cao**: 
            - MOTA (Multiple Object Tracking Accuracy) đạt 33.4%, nằm trong top các phương pháp tốt nhất
            - MOTP (Multiple Object Tracking Precision) đạt 72.1%, cho thấy độ chính xác về vị trí rất tốt
        
        - **Hiệu quả trong xử lý thời gian thực**:
            - Là phương pháp Online (xử lý theo thời gian thực)
            - Hiệu suất cạnh tranh với cả các phương pháp Batch (xử lý offline)
        
        - **Ưu điểm nổi bật**:
            - FAF (False Alarm per Frame) thấp: chỉ 1.3%, giảm thiểu cảnh báo sai
            - ML (Mostly Lost) thấp nhất: 30.9%, cho thấy khả năng duy trì track tốt
            - FP (False Positives) thấp: 7318, ít nhận diện sai
            - FN (False Negatives) ở mức tốt: 32615, cân bằng giữa bỏ sót và nhận diện sai
    """)

    # 1.2. Chi tiết thuật toán
    st.subheader("1.2. Chi tiết thuật toán")
    
    # Hiển thị hình ảnh minh họa tổng quan về SORT
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('SORT_process/image.png', caption='Sơ đồ tổng quan thuật toán SORT')
    
    st.markdown("""
    ### Thuật toán SORT hoạt động qua các bước chính:

    #### 1. Object Detection
    - Sử dụng một detector bên ngoài (như YOLO, SSD, Faster R-CNN) để phát hiện đối tượng trong mỗi frame
    - Mỗi detection bao gồm bounding box và điểm số tin cậy
    - Detector có thể được tùy chọn dựa trên yêu cầu về tốc độ và độ chính xác
    - Output của detector sẽ là danh sách các bounding box với format [x1, y1, x2, y2, score]

    #### 2. State Estimation với Kalman Filter
    - Dự đoán vị trí của đối tượng trong frame tiếp theo
    - Sử dụng mô hình chuyển động tuyến tính
    - State vector bao gồm 7 thành phần: [u, v, s, r, u̇, v̇, ṡ]
        - u, v: tọa độ tâm bbox
        - s: diện tích
        - r: tỷ lệ khung hình
        - u̇, v̇, ṡ: vận tốc tương ứng
    - Kalman Filter thực hiện hai bước:
        - Predict: Dự đoán state mới dựa trên mô hình chuyển động
        - Update: Cập nhật state dựa trên measurement mới

    #### 3. Data Association với Hungarian Algorithm
    - Liên kết các detection mới với các track hiện có
    - Sử dụng IoU (Intersection over Union) làm metric đo độ tương đồng
    - Quy trình association:
        1. Tính ma trận IoU giữa tất cả các cặp detection-track
        2. Áp dụng Hungarian Algorithm để tìm assignment tối ưu
        3. Lọc bỏ các assignment có IoU thấp hơn ngưỡng
    - Kết quả cho ra ba tập:
        - Matched tracks: Các cặp detection-track được ghép
        - Unmatched detections: Detections không được ghép với track nào
        - Unmatched tracks: Tracks không được ghép với detection nào

    #### 4. Track Management
    - Quản lý vòng đời của các track trong hệ thống
    - Các hoạt động chính:
        1. Khởi tạo track mới:
            - Tạo từ unmatched detections
            - Khởi tạo Kalman Filter với state vector ban đầu
        2. Cập nhật track hiện có:
            - Update Kalman Filter với matched detections
            - Cập nhật số frame liên tiếp track được/mất detection
        3. Xóa track:
            - Xóa track khi không được cập nhật trong max_age frames
            - Hoặc track có chất lượng thấp (ít hits)

    ### Quy trình hoạt động tổng thể:
    1. Nhận frame mới từ video input
    2. Thực hiện object detection trên frame
    3. Dự đoán vị trí mới cho các track hiện có (Kalman prediction)
    4. Liên kết detections với tracks (Data association)
    5. Cập nhật tracks phù hợp với detection mới
    6. Tạo tracks mới cho unmatched detections
    7. Xóa tracks không được cập nhật trong thời gian dài
    8. Trả về kết quả tracking cho frame hiện tại

    Mỗi bước trong thuật toán đều được thiết kế để đơn giản và hiệu quả, tập trung vào việc xử lý realtime. 
    Việc sử dụng Kalman Filter cho prediction và Hungarian Algorithm cho data association tạo nên một giải pháp 
    tracking đơn giản nhưng mạnh mẽ.
    """)

    # 1.3. Ví dụ minh họa
    st.subheader("1.3. Ví dụ minh họa")
    
    st.markdown("""
    ### Demo SORT Algorithm
    
    Dưới đây là ví dụ minh họa việc áp dụng thuật toán SORT để theo dõi đối tượng trong video.
    Video demo thể hiện các bước trong quy trình hoạt động của SORT:
    
    - Phát hiện đối tượng bằng YOLOv8
    - Tracking đối tượng qua các frame
    - Gán ID và theo dõi quỹ đạo chuyển động
    """)
    
    # Tạo 2 cột để căn giữa video
    col1, col2, col3 = st.columns([1,10,1])
    
    with col2:
        # Hiển thị video demo trực tiếp từ file
        st.video('SORT_process/output_tracked.mp4')  # Thay đổi tên file video của bạn
    
    st.markdown("""
    ### Giải thích kết quả:
    
    - **Bounding box màu**: Khung theo dõi đối tượng
    - **ID**: Số định danh duy nhất cho mỗi đối tượng
    - **Tọa độ**: Vị trí tâm của đối tượng
    - **Số lượng**: Tổng số đối tượng đang được theo dõi
    """)
    
    # 3. Thảo luận về các trường hợp thách thức
    st.header("3. Thảo luận về các trường hợp thách thức")
    
    # Tạo tabs cho các trường hợp
    tabs = st.tabs(["Background Clutters", "Illumination Variations", "Occlusion",])
    
    # Tab Background Clutters
    with tabs[0]:
        st.subheader("Background Clutters (Nền phức tạp)")
        
        # Tạo 2 cột cho video
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Video gốc")
            st.video('SORT_process/walking_output.mp4')
            
        with col2:
            st.markdown("### SORT tracking")
            st.video('SORT_process/output_tracked_walking.mp4')
            
        st.markdown("### Nhận xét khả năng tracking")
        st.markdown("""
        - **Thách thức**:
            - Nền phức tạp với nhiều đối tượng tương tự gây nhiễu nghiêm trọng
            - Sự tương đồng cao giữa đối tượng và background làm mất đặc trưng phân biệt
            - Nhiễu visual từ background gây ảnh hưởng lớn đến quá trình detection
            - Các đối tượng trong nền có thể được nhận diện nhầm là đối tượng cần track
            
        - **Đánh giá hiệu quả**:
            - SORT hoạt động kém hiệu quả trong môi trường background clutters
            - Detector thường xuyên bị nhầm lẫn, dẫn đến nhiều false positives
            - Kalman Filter không thể dự đoán chính xác do measurements nhiễu
            - Tracking thường xuyên bị gián đoạn và mất dấu đối tượng
            - Hiệu suất giảm mạnh khi đối tượng di chuyển qua vùng nền phức tạp
            - Tỷ lệ ID switches tăng cao do nhầm lẫn giữa đối tượng và nền
        """)
    
    # Tab Illumination Variations
    with tabs[1]:
        st.subheader("Illumination Variations (Thay đổi ánh sáng)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Video gốc")
            st.video('SORT_process/car4_output.mp4')
            
        with col2:
            st.markdown("### SORT tracking")
            st.video('SORT_process/output_tracked_car4.mp4')
            
        st.markdown("### Nhận xét khả năng tracking")
        st.markdown("""
        - **Thách thức từ Illumination Variations**:
            - Ngay cả những thay đổi nhỏ nhất về ánh sáng cũng gây ảnh hưởng nghiêm trọng
            - Sự thay đổi ánh sáng làm mất hoàn toàn khả năng theo dõi
            - Đối tượng bị biến mất khỏi tracking ngay khi có dao động về độ sáng
            - Detector hoàn toàn mất khả năng phát hiện trong điều kiện ánh sáng thay đổi
            
        - **Đánh giá hiệu quả**:
            - SORT thất bại hoàn toàn trong việc duy trì tracking khi có thay đổi ánh sáng
            - Detector không thể phát hiện đối tượng ngay cả với thay đổi ánh sáng nhỏ
            - Track bị mất ngay lập tức khi đối tượng di chuyển qua vùng có độ sáng khác
            - Không có khả năng khôi phục tracking sau khi mất do thay đổi ánh sáng
            - Tỷ lệ ID switches tăng vọt do liên tục mất và tạo track mới
        """)
    
    # Tab Occlusion
    with tabs[2]:
        st.subheader("Occlusion (Che khuất)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Video gốc")
            st.video('SORT_process/jogging_output.mp4')
            
        with col2:
            st.markdown("### SORT tracking")
            st.video('SORT_process/output_tracked_jogging.mp4')
            
        st.markdown("### Nhận xét khả năng tracking")
        st.markdown("""
        - **Thách thức trong Occlusion**:
            - Đối tượng bị che khuất một phần hoặc hoàn toàn
            - Mất hoàn toàn thông tin tracking khi bị occlusion
            - Không có cơ chế ghi nhớ đặc trưng của đối tượng
            - Đặc biệt khó khăn khi nhiều đối tượng che khuất lẫn nhau
            
        - **Đánh giá hiệu quả**:
            - SORT luôn mất track ngay khi đối tượng bị che khuất
            - Khi đối tượng xuất hiện lại, SORT sẽ:
                + Tạo một ID hoàn toàn mới
                + Coi đó là một đối tượng khác
                + Không nhận ra đây là đối tượng đã track trước đó
            - Thiếu hoàn toàn khả năng re-identification dẫn đến:
                + ID switching xảy ra thường xuyên
                + Một đối tượng có thể nhận nhiều ID khác nhau
                + Thống kê tracking bị sai lệch nghiêm trọng
        """)


if __name__ == "__main__":
    main()