import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import os

MODEL_FILENAME = "pixelrnn_best_model.pth"
MODEL_DIR = "outputs_new"
IMAGE_MIME_TYPE = "image/png"
IMAGE_SIZE = 64

st.set_page_config(
    page_title="PixelRNN Image Completion",
    page_icon="🎨",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f5f5f5;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    text-align: center;
    color: #fff;
}
hr {
    border: 1px solid rgba(255,255,255,0.3);
}
div[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.06);
    border: 2px dashed rgba(255,255,255,0.3);
    border-radius: 10px;
    padding: 1.5rem;
}
.stButton>button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.5rem 1rem !important;
    border: 2px solid #4A90E2 !important;
    transition: 0.3s ease-in-out !important;
}
.stButton>button:hover {
    transform: scale(1.05) !important;
    background-color: #f0f0f0 !important;
    color: #000000 !important;
}
.footer {
    text-align: center;
    font-size: 0.85rem;
    color: #ccc;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        from pixelrnn import PixelRNN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PixelRNN().to(device)

        ckpt_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            st.success(f"✅ Model loaded successfully ({ckpt_path})")
            return model, device, True
        else:
            st.error(f"❌ Model file not found: {ckpt_path}")
            return None, device, False
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, torch.device("cpu"), False


model, device, model_loaded = load_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
to_pil = transforms.ToPILImage()

st.markdown("<h1>🧠 PixelRNN Image Completion</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;'>
Upload an occluded image and let <b>PixelRNN</b> reconstruct the missing regions.<br>
Compare the input and output side by side.
</p>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Occluded Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model_loaded and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    original_size = image.size

    max_size = 350
    if original_size[0] > original_size[1]:
        display_width = min(max_size, original_size[0])
        display_height = int(display_width * original_size[1] / original_size[0])
    else:
        display_height = min(max_size, original_size[1])
        display_width = int(display_height * original_size[0] / original_size[1])
    display_size = (display_width, display_height)
    display_image = image.resize(display_size, Image.Resampling.LANCZOS)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("✨ Reconstructing image... Please wait."):
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.clamp(output, 0, 1)

    reconstructed = to_pil(output.squeeze().cpu()).resize(display_size, Image.Resampling.LANCZOS)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align:center;'>Original Input</h3>", unsafe_allow_html=True)
        st.image(display_image)
    with col2:
        st.markdown("<h3 style='text-align:center;'>AI Reconstructed</h3>", unsafe_allow_html=True)
        st.image(reconstructed)

    st.markdown("### 📥 Download Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        full_res = to_pil(output.squeeze().cpu()).resize(original_size, Image.Resampling.LANCZOS)
        buf_full = io.BytesIO()
        full_res.save(buf_full, format="PNG")
        st.download_button(
            label="Full Resolution",
            data=buf_full.getvalue(),
            file_name=f"pixelrnn_full_{original_size[0]}x{original_size[1]}.png",
            mime=IMAGE_MIME_TYPE,
        )

    with col2:
        buf_disp = io.BytesIO()
        reconstructed.save(buf_disp, format="PNG")
        st.download_button(
            label="Preview Size",
            data=buf_disp.getvalue(),
            file_name="pixelrnn_preview.png",
            mime=IMAGE_MIME_TYPE,
        )

    with col3:
        comparison = Image.new("RGB", (display_size[0] * 2, display_size[1]))
        comparison.paste(display_image, (0, 0))
        comparison.paste(reconstructed, (display_size[0], 0))
        buf_comp = io.BytesIO()
        comparison.save(buf_comp, format="PNG")
        st.download_button(
            label="Before & After",
            data=buf_comp.getvalue(),
            file_name="pixelrnn_comparison.png",
            mime=IMAGE_MIME_TYPE,
        )

elif uploaded_file and not model_loaded:
    st.error("❌ Model not available. Please ensure the model file exists in the outputs_new folder.")

elif not uploaded_file:
    st.info("📸 Please upload an image to get started!")
    st.markdown("### How It Works")
    st.markdown("""
    - Upload an image with missing or damaged regions  
    - PixelRNN processes it row by row to predict missing pixels  
    - AI reconstructs the occluded regions  
    - Download your restored image or compare results  
    """)
    st.markdown("### Pro Tips")
    st.markdown("""
    - Clear occlusions or masks work best  
    - Higher-quality images yield better completion  
    - Works best for structured scenes (faces, objects, nature)  
    """)

st.markdown("""
<div class='footer'>
    <p>🧩 PixelRNN Image Completion • Built with PyTorch & Streamlit</p>
    <p>Crafted by <a href="https://github.com/MuhammadMaaz7" target="_blank">Muhammad Maaz</a> • Open Source on GitHub</p>
</div>
""", unsafe_allow_html=True)
