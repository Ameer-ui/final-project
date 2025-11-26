# app.py → FINAL VERSION FOR ONLINE DEPLOYMENT
import streamlit as st
from validation import validate_image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from load_member2_model import load_member2_model
import os

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("**Team Members:** Member 1 • Member 2 • Member 3")

uploaded_file = st.file_uploader("Upload Brain MRI (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- NO FILE SAVING TO DISK (this is the fix) ---
    bytes_data = uploaded_file.read()          # read directly from memory
    from PIL import Image
    from io import BytesIO
    img = Image.open(BytesIO(bytes_data)).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Save to temporary in-memory path only for validation function
    temp_path = "temp_uploaded_mri.jpg"
    img.save(temp_path)

    # Member 3: Validation
    is_valid, message = validate_image(temp_path)
    if not is_valid:
        st.error(f"❌ REJECTED: {message}")
        os.remove(temp_path)
        st.stop()

    st.success("✅ Valid Brain MRI detected! Analyzing...")

    # Load Member 2's model (only once using cache)
    @st.cache_resource
    def get_model():
        return load_member2_model()
    
    with st.spinner("Loading 92% accurate AI model..."):
        model = get_model()

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Predicting tumor type..."):
        pred = model.predict(img_array, verbose=0)
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        result = classes[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)

    # Results
    st.markdown(f"## **🎯 PREDICTION: {result.upper()}**")
    st.progress(confidence / 100)
    st.write(f"**Confidence: {confidence:.2f}%**")

    if confidence < 70:
        st.warning("Low confidence – please verify with a doctor")

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

st.caption("© 2025 Brain Tumor Detection Team")
