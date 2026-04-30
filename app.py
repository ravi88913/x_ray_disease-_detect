import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="X-ray AI", layout="centered")

st.markdown("<h1 style='text-align: center;'>🩺 X-ray Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-based Chest X-ray Analysis</p>", unsafe_allow_html=True)

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "xray_model.keras",
        custom_objects={'loss': focal_loss()}
    )
    return model

model = load_model()

labels = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia","No Finding"
]

st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    image = image.resize((224,224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    threshold = 0.4

    found = False
    for i in range(len(labels)):
        if preds[i] > threshold:
            found = True
            st.write(f"**{labels[i]}** : {preds[i]*100:.2f}%")

    if not found:
        st.info("No strong disease detected")

    st.markdown("---")
    st.subheader("🏆 Top 3 Predictions")

    top_indices = preds.argsort()[-3:][::-1]

    for i in top_indices:
        st.write(f"**{labels[i]}**")
        st.progress(float(preds[i]))
        st.write(f"{preds[i]*100:.2f}%")
        st.markdown("")

    st.markdown("---")

    if preds[top_indices[0]] > 0.6:
        st.success("High Confidence Prediction")
    else:
        st.warning("Low Confidence Prediction")
