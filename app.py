import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

MODEL_PATH = 'models/rice_model_fast.h5' 
IMAGE_SIZE = (64, 64)
CLASS_NAMES = [
    'Class 01', 'Class 02', 'Class 03', 'Class 04', 'Class 05', 
    'Class 06', 'Class 07', 'Class 08', 'Class 09', 'Class 10'
] 

@st.cache_resource
def load_rice_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Lỗi tải mô hình: Vui lòng kiểm tra file {MODEL_PATH} đã được đẩy lên GitHub chưa.")
        st.code(f"Chi tiết lỗi: {e}")
        return None

def predict_image(image_file, model, class_names, img_size):
    try:
        img = Image.open(image_file).convert('RGB').resize(img_size)
        
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        img_array_expanded = img_array_expanded / 255.0 

        predictions = model.predict(img_array_expanded)
        
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return class_names[predicted_class_index], confidence

    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý ảnh hoặc dự đoán: {e}")
        return "Lỗi xử lý", 0.0

model = load_rice_model()

st.set_page_config(page_title="Phân Loại Gạo CNN", layout="centered")

st.title("🍚 Ứng Dụng Phân Loại Gạo Tự Động (Mô hình CNN)")
st.caption(f"Kiến trúc CNN đã huấn luyện, Kích thước ảnh: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels.")

if model is not None:
    uploaded_file = st.file_uploader("Tải lên ảnh hạt gạo để phân loại...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Ảnh Gạo Đã Tải Lên', width=200)
        
        if st.button('Phân Loại Hạt Gạo'):
            with st.spinner('Đang xử lý và dự đoán...'):
                
                predicted_name, confidence = predict_image(uploaded_file, model, CLASS_NAMES, IMAGE_SIZE)
                
                if predicted_name != "Lỗi xử lý":
                    st.success(f"✅ Dự đoán Thành Công!")
                    st.markdown("---")
                    st.markdown(f"**Kết quả Phân loại:** <span style='color:green; font-size: 20px;'>{predicted_name}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Độ tin cậy:** **{confidence:.2%}**")
                    st.markdown("---")
else:
    st.error("Ứng dụng không thể khởi động vì mô hình không được tải. Vui lòng kiểm tra file rice_model_fast.h5 và log deploy.")
