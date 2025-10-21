import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ===================================================================
# 1. CÁC THAM SỐ CẤU HÌNH VÀ TÊN LỚP
# ===================================================================

# Tên file mô hình đã được đẩy lên GitHub
MODEL_PATH = 'rice_model_fast.h5' 
IMAGE_SIZE = (64, 64) # Kích thước ảnh đầu vào đã dùng khi huấn luyện
# Tên các lớp theo yêu cầu của bạn (10 lớp, thứ tự index 0-9 tương ứng Class 01-10)
CLASS_NAMES = [
    'Class 01', 'Class 02', 'Class 03', 'Class 04', 'Class 05', 
    'Class 06', 'Class 07', 'Class 08', 'Class 09', 'Class 10'
] 

# ===================================================================
# 2. HÀM TẢI MÔ HÌNH VÀ DỰ ĐOÁN
# ===================================================================

# Sử dụng @st.cache_resource để chỉ tải mô hình một lần duy nhất
@st.cache_resource
def load_rice_model():
    """Tải mô hình Keras đã lưu."""
    try:
        # Tải mô hình từ file .h5
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        # Xử lý lỗi nếu file mô hình không tìm thấy hoặc không tải được (Quan trọng cho Deploy)
        st.error(f"Lỗi tải mô hình: Vui lòng kiểm tra file {MODEL_PATH} đã được đẩy lên GitHub chưa.")
        st.code(f"Chi tiết lỗi: {e}")
        return None

def predict_image(image_file, model, class_names, img_size):
    """Xử lý ảnh đầu vào và dự đoán lớp gạo."""
    try:
        # Đọc file ảnh, chuyển sang RGB và resize về kích thước (64, 64)
        img = Image.open(image_file).convert('RGB').resize(img_size)
        
        # Chuyển ảnh thành mảng Numpy (định dạng TensorFlow)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Thêm chiều Batch (batch size = 1)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # CHUẨN HÓA: Rescale ảnh về (0-1), BẮT BUỘC phải khớp với quá trình huấn luyện
        img_array_expanded = img_array_expanded / 255.0 

        # Dự đoán
        predictions = model.predict(img_array_expanded)
        
        # Lấy lớp có xác suất cao nhất
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return class_names[predicted_class_index], confidence

    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý ảnh hoặc dự đoán: {e}")
        return "Lỗi xử lý", 0.0

# Tải mô hình
model = load_rice_model()

# ===================================================================
# 3. GIAO DIỆN STREAMLIT
# ===================================================================

st.set_page_config(page_title="Phân Loại Gạo CNN", layout="centered")

st.title("🍚 Ứng Dụng Phân Loại Gạo Tự Động (Mô hình CNN)")
st.caption(f"Kiến trúc CNN đã huấn luyện, Kích thước ảnh: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels.")

# Kiểm tra nếu mô hình tải thành công
if model is not None:
    uploaded_file = st.file_uploader("Tải lên ảnh hạt gạo để phân loại...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Ảnh Gạo Đã Tải Lên', width=200)
        
        # Nút bấm chạy dự đoán
        if st.button('Phân Loại Hạt Gạo'):
            with st.spinner('Đang xử lý và dự đoán...'):
                
                # Gọi hàm dự đoán
                predicted_name, confidence = predict_image(uploaded_file, model, CLASS_NAMES, IMAGE_SIZE)
                
                if predicted_name != "Lỗi xử lý":
                    st.success(f"✅ Dự đoán Thành công!")
                    st.markdown("---")
                    st.markdown(f"**Kết quả Phân loại:** <span style='color:green; font-size: 20px;'>{predicted_name}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Độ tin cậy:** **{confidence:.2%}**")
                    st.markdown("---")
                # else: Lỗi đã được in ra trong hàm predict_image
else:
    st.error("Ứng dụng không thể khởi động vì mô hình không được tải. Vui lòng kiểm tra file rice_model_fast.h5 và log deploy.")