import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

def main():
    st.title("KCF (Kernelized Correlation Filter) Object Tracking")

    # Phần 1
    st.header("Phần 1: Nguyên lý hoạt động")
    
    st.markdown("""
    ### A. Tổng quan
    `KCF` (`Kernelized Correlation Filter`) là thuật toán `tracking` dựa trên `correlation filter` với `kernel trick`. Điểm đặc biệt của `KCF` là sử dụng tính chất tuần hoàn của ma trận để tăng tốc độ tính toán và áp dụng `kernel trick` để xử lý trong không gian đặc trưng phi tuyến.

    ### B. Các thành phần chính:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        1. **Vùng đối tượng (`ROI`):**
           - Vùng chứa đối tượng cần theo dõi
           - Được chọn từ `frame` đầu tiên
           - Là mẫu `positive sample` ban đầu
        
        2. **Đặc trưng `HOG`:**
           - Mô tả `gradient` của ảnh
           - Bất biến với thay đổi ánh sáng
           - Giúp nhận dạng đặc điểm của đối tượng
        """)

    with col2:
        st.markdown("""
        3. **Kernel Correlation:**
           - Đo độ tương tự giữa các mẫu
           - Sử dụng `Gaussian kernel`
           - Cho phép xử lý phi tuyến
        
        4. **Response Map:**
           - Biểu diễn xác suất vị trí đối tượng
           - Giá trị cao thể hiện khả năng xuất hiện
           - Dùng để xác định vị trí mới
        """)

    st.markdown("""
    ### C. Thuật toán
    #### C.1. Khởi tạo (`Frame` đầu tiên):
    1. **Chọn vùng đối tượng:**
       - Người dùng chọn `ROI` (`Region of Interest`) bằng `cv2.selectROI`
       - Trích xuất đặc trưng `HOG` từ `ROI`: `x = hog(ROI)`
       - `HOG features` giúp mô tả `gradient` của ảnh, bất biến với thay đổi ánh sáng
    
    2. **Tạo nhãn mong muốn:**
       - Tạo `Gaussian label` y (`ground truth`): `y = exp(-||x-x_c||²/(2σ²))`
       - `Label` y có giá trị cao nhất tại tâm và giảm dần ra xung quanh
       - Biến đổi `Fourier`: `F(y)` để chuyển sang miền tần số
    
    3. **Học bộ lọc ban đầu:**
       - Áp dụng `kernel`: `k(x,x') = exp(-||x-x'||²/(2σ²))`
       - `Kernel trick` cho phép học trong không gian đặc trưng phi tuyến
       - Tính `correlation filter`: `F(f) = F(y) ⊘ F(x)`
       - Lưu trữ mô hình ban đầu làm `template`

    #### C.2. Tracking (Các frame tiếp theo):
    1. **Trích xuất đặc trưng:**
       - Xác định vùng tìm kiếm (`search area`) z quanh vị trí cũ
       - `Search area` lớn hơn `ROI` để đảm bảo bắt được chuyển động
       - Trích xuất `HOG features` từ z theo cùng cách với `template`
    
    2. **Tính toán `correlation`:**
       - Tính `kernel correlation`: `k_xz = k(x,z)`
       - `Kernel` đo độ tương tự giữa `template` và vùng tìm kiếm
       - Biến đổi `Fourier`: `F(k_xz)` để tính toán nhanh trong miền tần số
    
    3. **Xác định vị trí:**
       - Tính `response map`: `g(z) = F⁻¹(F(k_xz) ⊙ F(α))`
       - `Response map` cho biết độ tương tự tại mỗi vị trí
       - Tìm vị trí có `response` cao nhất: `(x,y) = argmax(g(z))`
       - Độ tin cậy của `tracking` dựa vào giá trị `response` tại vị trí tìm được
    
    4. **Cập nhật mô hình:**
       - Cập nhật mẫu x với vị trí mới: `x_new = (1-η)x + ηx_t`
       - η là `learning rate`, điều chỉnh tốc độ cập nhật mô hình
       - Cập nhật bộ lọc f để thích nghi với thay đổi của đối tượng
       - Việc cập nhật giúp `tracker` thích nghi với biến dạng và thay đổi góc nhìn

    #### C.3. Các tham số trong `implementation`:
    1. **Tham số video:**
    - Frame Width: `frame_width = 640`
    - Frame Height: `frame_height = 480`
    - FPS: `fps = 30`
    - Video Format: `fourcc = 'mp4v'`

    2. **Tham số hiển thị:**
    - Tracking Color: `(0, 255, 0)` (`Green`)
    - Lost Color: `(0, 0, 255)` (`Red`)
    - Line Thickness: `2`
    - Font: `cv2.FONT_HERSHEY_SIMPLEX`
    - Font Scale: `1.0`

    """)

    # Chèn hình ảnh sau phần C
    st.image("new_videos/3.jpg", 
             caption="Sơ đồ tổng quan thuật toán KCF Tracking",
             use_column_width=True)

    # Giới thiệu chung cho cả 2 video
    st.markdown("""
    ### D. Ví dụ minh họa
    Video dưới đây thể hiện khả năng theo dõi chuyển động của KCF:
    - Đối tượng: Người đi bộ trên đường
    - Điều kiện: Nền đơn giản, ánh sáng tốt
    """)

    # Hiển thị 2 video trong 2 cột
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Video 1**")
        video_file = open('new_videos/output.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    with col2:
        st.markdown("**Video 2**")
        video_file = open('new_videos/output1.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    # Chuyển sang Phần 3
    st.header("Phần 3: Thảo luận về các trường hợp thách thức")
    tabs = st.tabs(["Background Clutters", "Illumination Variations", "Occlusion", "Fast Motion"])

    with tabs[0]:
        st.subheader("Background Clutters (Nền phức tạp)")
        # Tạo 2 cột để hiển thị video
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Video gốc**")
            video_path = 'new_videos/walking_output.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                # Debug info
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        with col2:
            st.markdown("**Video có áp dụng KCF tracking**")
            video_path = 'new_videos/walking_result.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                # Debug info
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        # Phần nhận xét khả năng tracking
        # Phần nhận xét khả năng tracking
        st.markdown("""
        #### Nhận xét khả năng tracking
        1. **Điểm mạnh quan sát được:**
           - KCF duy trì tracking tốt trong vài giây đầu (0:00-0:02) khi xe di chuyển trên nền đường đơn giản
           - HOG features giúp phân biệt được đặc trưng của đối tượng với nền trong thời gian ngắn ban đầu
           - Thuật toán vẫn duy trì được tracking trong một số trường hợp nền đơn giản
           - HOG features giúp phân biệt được đối tượng với nền trong điều kiện ánh sáng tốt
        
        2. **Hạn chế rõ rệt:**
           - Mất hoàn toàn khả năng tracking từ rất sớm (khoảng giây thứ 3)
           - Bounding box đứng yên tại chỗ trong khi đối tượng tiếp tục di chuyển (0:03-end)
           - Không có khả năng phục hồi tracking sau khi mất dấu
           - Dễ bị nhiễu khi có nhiều đối tượng tương tự trong nền
           - Có thể mất dấu khi đối tượng di chuyển qua vùng có kết cấu phức tạp
           - Hiệu suất giảm khi nền có màu sắc tương tự đối tượng tracking
        """)
        
    with tabs[1]:
        st.subheader("Illumination Variations (Thay đổi ánh sáng)")
        # Tạo 2 cột để hiển thị video
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Video gốc**")
            video_path = 'new_videos/car4_output.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        with col2:
            st.markdown("**Video có áp dụng KCF tracking**")
            video_path = 'new_videos/car4_result.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        st.markdown("""
        #### Nhận xét khả năng tracking
        1. **Điểm mạnh quan sát được:**
           - KCF duy trì tracking tốt khi xe di chuyển trên nền đường đơn giản (0:00-0:03)
           - Thuật toán có khả năng phân biệt đối tượng với các xe khác có màu sắc khác biệt (0stre:04-0:06)
           - HOG features giúp phân biệt được đặc trưng của đối tượng với nền đường và các vạch kẻ (0:07-0:09)
           - Thuật toán vẫn duy trì được tracking trong một số trường hợp nền đơn giản
           - HOG features giúp phân biệt được đối tượng với nền trong điều kiện ánh sáng tốt
        
        2. **Hạn chế rõ rệt:**
           - Tại thời điểm 0:10-0:12, khi có nhiều xe cùng màu xuất hiện trong khung hình, bounding box có xu hướng bị nhiễu
           - Tracking bị ảnh hưởng khi đối tượng di chuyển qua các vùng có họa tiết phức tạp trên mặt đường (0:13-0:15)
           - Độ chính xác giảm khi có nhiều đối tượng tương tự (về màu sắc và kích thước) xuất hiện gần đối tượng đang theo dõi (0:16-0:18)
           - Khó khăn trong việc duy trì tracking khi nền có độ tương phản cao hoặc có nhiều chi tiết phức tạp
           - Dễ bị nhiễu khi có nhiều đối tượng tương tự trong nền
           - Có thể mất dấu khi đối tượng di chuyển qua vùng có kết cấu phức tạp
           - Hiệu suất giảm khi nền có màu sắc tương tự đối tượng tracking
        """)

    with tabs[2]:
        st.subheader("Occlusion (Che khuất)")
        # Tạo 2 cột để hiển thị video
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Video gốc**")
            video_path = 'new_videos/jogging_output.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        with col2:
            st.markdown("**Video có áp dụng KCF tracking**")
            video_path = 'new_videos/jogging_result.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        st.markdown("""
        #### Nhận xét khả năng tracking
        1. **Điểm mạnh quan sát được:**
           - KCF duy trì tracking tốt khi người chạy bộ chưa bị che khuất (0:00-0:02)
           - Thuật toán vẫn theo dõi được khi người chạy bộ bị che khuất một phần bởi cây cột (0:03-0:04)
           - Đặc biệt, sau khi bị che khuất hoàn toàn bởi cột (0:05-0:06), KCF có khả năng phục hồi và bắt lại được đối tượng khi xuất hiện ở phía bên kia cột (0:07)
           - Bounding box tiếp tục bám sát đối tượng sau khi phục hồi tracking

        2. **Hạn chế rõ rệt:**
           - Trong thời điểm bị che khuất hoàn toàn (0:05-0:06), bounding box tạm thời mất dấu đối tượng
           - Có độ trễ nhỏ trong việc phục hồi tracking sau khi đối tượng xuất hiện trở lại
           - Độ chính xác của bounding box giảm đi một chút ngay sau khi phục hồi tracking
        """)

    with tabs[3]:
        st.subheader("Fast Motion (Chuyển động nhanh)")
        # Tạo 2 cột để hiển thị video
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Video gốc**")
            video_path = 'new_videos/jumping_output.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        with col2:
            st.markdown("**Video có áp dụng KCF tracking**")
            video_path = 'new_videos/jumping_result.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        st.markdown("""
        #### Nhận xét khả năng tracking
        1. **Điểm mạnh quan sát được:**
           - KCF chỉ duy trì được tracking trong vài giây đầu (0:00-0:02) khi người chạy bộ bắt đầu di chuyển
           - Bounding box ban đầu bám được đối tượng trong khoảng thời gian ngắn với chuyển động chậm
           - Tracking tốt với chuyển động tốc độ vừa phải
           - Tính toán nhanh nhờ FFT trong miền tần số

        2. **Hạn chế rõ rệt:**
           - Ngay khi người chạy bộ bắt đầu tăng tốc (0:03), thuật toán hoàn toàn mất khả năng tracking
           - Bounding box đứng yên tại chỗ trong khi đối tượng đã di chuyển ra khỏi vùng tracking (0:03-end)
           - Không có khả năng phục hồi tracking sau khi mất dấu do chuyển động nhanh
           - Thuật toán thể hiện rõ hạn chế trong việc xử lý các chuyển động có tốc độ cao
           - Mất dấu khi đối tượng di chuyển quá nhanh
           - Vùng tìm kiếm có kích thước cố định
           - Không xử lý được chuyển động đột ngột
        """)

if __name__ == '__main__':
    main()