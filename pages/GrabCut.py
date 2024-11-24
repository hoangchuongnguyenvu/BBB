from PIL import Image
import streamlit as st
from components.grabcut import (
    display_form_draw,
    display_st_canvas,
    init_session_state,
    process_grabcut,
)
from services.grabcut.ultis import get_object_from_st_canvas

init_session_state()

st.set_page_config(
    page_title="Ứng dụng tách nền bằng thuật toán GrabCut",
    #page_icon=Image.open("./public/images/logo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Thêm CSS để xóa tất cả khung viền
st.markdown("""
    <style>
    /* Xóa border cho tất cả container */
    .element-container {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Xóa border cho file uploader */
    .stFileUploader {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Xóa border cho selectbox */
    .stSelectbox {
        border: none !important;
    }
    
    /* Xóa border cho slider */
    .stSlider {
        border: none !important;
    }
    
    /* Xóa border cho tất cả các div */
    div[data-testid="stBlock"] {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Xóa border cho canvas */
    .canvas-container {
        border: none !important;
    }
    
    /* Xóa border cho image */
    div[data-testid="stImage"] {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Xóa border cho button */
    .stButton {
        border: none !important;
    }
    
    /* Thêm CSS mới để xóa khung cho phần hướng dẫn */
    div[data-testid="stMarkdown"] {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Xóa border cho tt cả các block */
    [data-testid="stBlock"] {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Xóa border cho tất cả các container */
    .block-container {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Xóa border cho các thẻ div chứa nội dung */
    div.stMarkdown {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Tạo slider với JavaScript để cập nhật giá trị gradient

# Thêm CSS để tùy chỉnh độ dày của thanh slider dựa trên giá trị

# Tạo slider và cập nhật CSS variable
# Thiết lập tiêu đề
st.title("ỨNG DỤNG TÁCH NỀN BẰNG THUẬT TOÁN GRABCUT")

with st.container():
    uploaded_image = st.file_uploader(
        ":material/image: Chọn hoặc kéo ảnh vào ô bên dưới", type=["jpg", "jpeg", "png"]
    )

if uploaded_image is not None:
    with st.container():
        # Phần hướng dẫn
        st.markdown(
            """
            <div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px;">
                <h3>🎯 Hướng dẫn sử dụng:</h3>
                <ol>
                    <li>Vẽ hình chữ nhật lên ảnh để chọn vùng cần tách nền.</li>
                    <li>Chọn chế độ vẽ và vẽ lên ảnh để chỉ định:
                        <ul>
                            <li>🟢 <b>Vùng giữ lại</b>: Vẽ màu xanh cho vùng chắc chắn giữ lại</li>
                            <li>🔴 <b>Vùng loại bỏ</b>: Vẽ màu đỏ cho vùng chắc chắn loại bỏ</li>
                        </ul>
                    </li>
                    <li>Ấn nút "Apply GrabCut" để xem kết quả.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Phần lưu ý (sử dụng markdown thuần túy)
        st.warning("""
        **Lưu ý:**
        - Vẽ càng chính xác, kết quả càng tốt
        - Có thể vẽ nhiều lần để điều chỉnh
        - Độ dày nét vẽ có thể thay đổi tùy ý
        """)

    with st.container():
        drawing_mode, stroke_width = display_form_draw()

    with st.container():
        cols = st.columns(2, gap="large")
        raw_image = Image.open(uploaded_image)

        with cols[0]:
            canvas_result = display_st_canvas(raw_image, drawing_mode, stroke_width)
            rects, true_fgs, true_bgs = get_object_from_st_canvas(canvas_result)

        if len(rects) < 1:
            st.session_state["result_grabcut"] = None
            st.session_state["final_mask"] = None
        elif len(rects) > 1:
            st.warning("Chỉ được chọn một vùng cần tách nền")
        else:
            with cols[0]:
                submit_btn = st.button("🎯 Apply GrabCut")

            if submit_btn:
                with st.spinner("Đang xử lý..."):
                    result = process_grabcut(
                        raw_image, canvas_result, rects, true_fgs, true_bgs
                    )
                    cols[1].image(result, channels="BGR", caption="Ảnh kết quả")
            elif st.session_state["result_grabcut"] is not None:
                cols[1].image(
                    st.session_state["result_grabcut"],
                    channels="BGR",
                    caption="Ảnh kết quả",
                )