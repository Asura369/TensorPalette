import io
import os
import time

import streamlit as st
from PIL import Image

from styleforge.engine import InferenceEngine

st.set_page_config(
    page_title="StyleForge",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.info("Local dev UI. Production app: see FastAPI + React frontend.")

# --- CSS Hacks ---
st.markdown("""
    <style>
        [data-testid="stFileUploader"] section > div:first-child + div {
            display: none;
        }
        [data-testid="stFileUploader"] section {
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🎨 StyleForge")
st.markdown("### High-Fidelity Neural Style Transfer Engine")
st.write("Transform photos into artistic styles using a production-ready Fast Neural Style Transfer engine.")
st.caption("Supported formats: JPG, PNG, JPEG.")

# Sidebar for controls
st.sidebar.header("Configuration")

# Style Selection
style_options = {
    "Starry Night": 0,
    "The Great Wave": 1,
    "Girl with a Pearl Earring": 2,
    "Composition VIII": 3,
    "Water Lilies": 4,
}
style_name = st.sidebar.selectbox("Select Art Style", list(style_options.keys()))
style_id = style_options[style_name]

# Quality Mode
quality_mode = st.sidebar.radio(
    "Processing Quality",
    ("Standard", "High Res"),
    help="Standard: Caps size at 1280px (Fast). High Res: Uses original upload quality (For High Res uploads)."
)

# Style Strength Slider
style_strength = st.sidebar.slider(
    "Style Strength",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="0.0 = Original Image, 1.0 = Full Style"
)

@st.cache_resource
def get_engine():
    return InferenceEngine()

engine = get_engine()

CIN_MODEL_PATH = "models/multistyle.pth"
NUM_STYLES = 5

if os.path.exists(CIN_MODEL_PATH):
    engine.load_cin("cin", CIN_MODEL_PATH, num_styles=NUM_STYLES)
else:
    st.sidebar.warning(f"CIN model not found at `{CIN_MODEL_PATH}`. Train or download first.")

def style_transfer(image, engine, style_id):
    return engine.stylize(image, "cin", style_id=style_id)

# --- Main Interface ---
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # Load Original
    content_image = Image.open(uploaded_file).convert('RGB')
    width, height = content_image.size

    with col1:
        st.subheader("Original Image")
        st.caption(f"Resolution: {width} x {height} px")
        st.image(content_image, width="stretch")

    with col2:
        st.subheader("Stylized Result")

        # Check cache logic
        current_state_id = f"{uploaded_file.file_id}_{style_name}_{style_strength}_{quality_mode}"
        has_result = ('last_result_id' in st.session_state and
                      st.session_state.last_result_id == current_state_id)

        # The Button
        if not has_result:
            st.info("Configure settings in sidebar, then click below.")
            if st.button("Stylize Image", type="primary", use_container_width=True):

                if "cin" in engine._models:
                    with st.status("Processing...", expanded=True) as status:

                        # 1. Resize Logic
                        max_dim = 1280
                        if quality_mode == "Standard" and max(width, height) > max_dim:
                            status.write("Standard Mode: Cap to 1280px...")
                            ratio = max_dim / max(width, height)
                            new_size = (int(width * ratio), int(height * ratio))
                            content_image_proc = content_image.resize(new_size, Image.Resampling.LANCZOS)
                        else:
                            if quality_mode == "High Res":
                                status.write(f"High Res Mode: Keeping {width}x{height}px...")
                            else:
                                status.write("Image is small enough. No resize needed.")
                            content_image_proc = content_image

                        # 2. Inference
                        status.write(f"Applying **{style_name}** style...")
                        start_time = time.time()
                        stylized_raw = style_transfer(content_image_proc, engine, style_id)
                        elapsed = time.time() - start_time

                        # 3. Blending
                        if style_strength < 1.0:
                            status.write("Mixing with original...")
                            if stylized_raw.size != content_image_proc.size:
                                stylized_raw = stylized_raw.resize(content_image_proc.size)
                            final_image = Image.blend(content_image_proc, stylized_raw, style_strength)
                        else:
                            final_image = stylized_raw

                        status.update(label=f"Done in {elapsed:.2f}s!", state="complete", expanded=False)

                    # Save to State
                    st.session_state.last_result_image = final_image
                    st.session_state.last_result_id = current_state_id
                    st.rerun()

        # If result exists, show it
        if has_result:
            final_image = st.session_state.last_result_image
            res_w, res_h = final_image.size
            st.caption(f"Resolution: {res_w} x {res_h} px")
            st.image(final_image, width="stretch")

            if st.button("Reset / New Style"):
                del st.session_state.last_result_id
                st.rerun()

            original_name = os.path.splitext(uploaded_file.name)[0]
            new_filename = f"{original_name}_{style_name}.jpg"
            buf = io.BytesIO()
            final_image.save(buf, format="JPEG", quality=95)

            st.download_button(
                label="Download Result",
                data=buf.getvalue(),
                file_name=new_filename,
                mime="image/jpeg"
            )

else:
    if 'last_result_id' in st.session_state:
        del st.session_state.last_result_id

    st.info("Upload an image on the left to get started.")

st.markdown("---")
st.markdown("Built with PyTorch & Streamlit | StyleForge")
