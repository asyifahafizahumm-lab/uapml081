import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
# Tambahkan import untuk preprocessing MobileNetV2 agar tidak error
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_preprocess
from pathlib import Path
import cv2
import traceback

# ========================
# KONFIGURASI DASHBOARD
# ========================
st.set_page_config(
    page_title="Klasifikasi Masker Wajah",
    layout="centered",
    page_icon="😷",
)

# CSS untuk tampilan
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 60%, #e8fff5 100%);
    }
    .title-container {
        background-color: #1f4b99;
        padding: 18px 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 16px;
    }
    .info-box {
        background-color: #ffffffaa;
        padding: 12px 16px;
        border-radius: 10px;
        border-left: 5px solid #1f4b99;
    }
    .footer {
        font-size: 11px;
        color: #666666;
        text-align: center;
        margin-top: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================
# KONFIGURASI MODEL
# ========================
IMG_SIZE = 128
CLASS_NAMES = ["with_mask", "without_mask"]
DISPLAY_LABEL = {
    "with_mask": "MEMAKAI MASKER",
    "without_mask": "TIDAK MEMAKAI MASKER",
}

@st.cache_resource
def load_model_cached(model_choice=None):
    base = Path(__file__).resolve().parent
    candidates = [
        base / "mobilenetv2_mask.h5",
        base / "vgg16_mask.h5",
        base / "base_cnn.h5",
    ]

    available = [p for p in candidates if p.exists()]

    if model_choice and model_choice != "Auto":
        chosen_path = base / model_choice
        if not chosen_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {model_choice}")
        model_path = chosen_path
    else:
        if not available:
            raise FileNotFoundError("Tidak ditemukan file model .h5 di folder src/")
        model_path = available[0]

    model = load_model(str(model_path))
    return model, model_path.name

def get_available_model_files():
    base = Path(__file__).resolve().parent
    return [p.name for p in (base / "*.h5").parent.glob("*.h5")]

# ========================
# FUNGSI PREDIKSI
# ========================
def run_prediction(image: Image.Image, input_size: int = IMG_SIZE, swap_labels: bool = False, model_choice: str | None = None):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        margin = int(0.25 * max(w, h))
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(img_cv.shape[1], x+w+margin), min(img_cv.shape[0], y+h+margin)
        face = img_cv[y1:y2, x1:x2]
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize((input_size, input_size))
    else:
        img = image.resize((input_size, input_size))

    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)

    model, model_name = load_model_cached(model_choice)
    model_name_l = model_name.lower()

    if "mobilenet" in model_name_l:
        arr = mn_preprocess(arr.astype("float32"))
        used_preprocess = "mobilenet_v2"
    elif "vgg" in model_name_l:
        from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
        arr = vgg_preprocess(arr.astype("float32"))
        used_preprocess = "vgg16"
    else:
        arr = arr.astype("float32") / 255.0
        used_preprocess = "scale_0_1"

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))

    labels = CLASS_NAMES.copy()
    if swap_labels:
        labels = labels[::-1]

    internal_label = labels[idx]
    display_label = DISPLAY_LABEL[internal_label]
    return internal_label, display_label, float(probs[idx]), probs, model_name, used_preprocess

# ========================
# TAMPILAN UTAMA
# ========================
def app():
    st.markdown('<div class="title-container"><h2>Klasifikasi Masker Wajah</h2></div>', unsafe_allow_html=True)

    available_models = get_available_model_files()
    selected_model = st.selectbox("Pilih model", ["Auto"] + available_models)

    uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)
        
        swap = st.checkbox("Tukar mapping kelas (jika terbalik)")
        
        if st.button("Prediksi"):
            try:
                with st.spinner("Memproses..."):
                    res = run_prediction(img, swap_labels=swap, model_choice=selected_model)
                    internal_label, display_label, conf, probs, model_name, used_preprocess = res

                if internal_label == "with_mask":
                    st.success(f"HASIL: {display_label} ({conf:.2%})")
                else:
                    st.error(f"HASIL: {display_label} ({conf:.2%})")
                
                st.info(f"Model: {model_name} | Preprocess: {used_preprocess}")
            except Exception as e:
                st.error(f"Error: {e}")
                st.text(traceback.format_exc())

if __name__ == "__main__":
    app()