import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
import os
import io
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformer import StyleTransformer
import utils

# Page Config
st.set_page_config(
    page_title="TensorPalette",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
st.title("🎨 TensorPalette")
st.markdown("### High-Fidelity Neural Style Transfer Engine")
st.write("Transform your photos into specific artistic styles using our production-ready Fast Neural Style Transfer architecture.")
st.caption("Supported formats: JPG, PNG, JPEG.")

# Sidebar for controls
st.sidebar.header("Configuration")

# Style Selection
style_name = st.sidebar.selectbox(
    "Select Art Style",
    ("Anime", "Sketch", "Oil", "Eastern")
)

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

# Model paths
STYLE_MODELS = {
    "Anime": "models/anime.pth",
    "Sketch": "models/sketch.pth",
    "Oil": "models/oil.pth",
    "Eastern": "models/eastern.pth"
}

model_path = STYLE_MODELS.get(style_name)

# Device Selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

st.sidebar.markdown(f"**Inference Device:** `{device}`")

@st.cache_resource
def load_model(path):
    model = StyleTransformer()
    if path and os.path.exists(path):
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        return None
    model.to(device)
    model.eval()
    return model

model = load_model(model_path)

if model is None:
    st.sidebar.warning(f"Model file `{model_path}` not found.")

def style_transfer(image, model):
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(content_image).cpu()

    img = output[0].clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(img)

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
                
                if model is not None:
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
                        status.write(f"🎨 Applying **{style_name}** style...")
                        start_time = time.time()
                        stylized_raw = style_transfer(content_image_proc, model)
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
st.markdown("Built with PyTorch & Streamlit | TensorPalette")
