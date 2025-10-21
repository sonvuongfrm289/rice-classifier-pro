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
        st.error(f"Loi tai mo hinh: Vui long kiem tra file {MODEL_PATH} da duoc day len GitHub chua.")
        st.code(f"Chi tiet loi: {e}")
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
        st.error(f"Loi trong qua trinh xu ly anh hoac du doan: {e}")
        return "Loi xu ly", 0.0

model = load_rice_model()

st.set_page_config(page_title="Phan Loai Gao CNN", layout="centered")

st.title("🍚 Ung Dung Phan Loai Gao Tu Dong (Mo hinh CNN)")
st.caption(f"Kien truc CNN da huan luyen, Kich thuoc anh: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels.")

if model is not None:
    uploaded_file = st.file_uploader("Tai len anh hat gao de phan loai...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Anh Gao Da Tai Len', width=200)
        
        if st.button('Phan Loai Hat Gao'):
            with st.spinner('Dang xu ly va du doan...'):
                
                predicted_name, confidence = predict_image(uploaded_file, model, CLASS_NAMES, IMAGE_SIZE)
                
                if predicted_name != "Loi xu ly":
                    st.success(f"✅ Du doan Thanh cong!")
                    st.markdown("---")
                    st.markdown(f"**Ket qua Phan loai:** <span style='color:green; font-size: 20px;'>{predicted_name}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Do tin cay:** **{confidence:.2%}**")
                    st.markdown("---")
else:
    st.error("Ung dung khong the khoi dong vi mo hinh khong duoc tai. Vui long kiem tra file rice_model_fast.h5 va log deploy.")
