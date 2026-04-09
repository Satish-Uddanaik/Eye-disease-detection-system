import tempfile
import os
from pathlib import Path
import streamlit as st
import cv2
import matplotlib.pyplot as plt

# Use legacy Keras deserialization for older HDF5 models.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Trained_Eye_disease_model_v2.keras"
OUTPUT_DIR = BASE_DIR / "output"

st.set_page_config(
    page_title="Human Eye Disease Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_THEME = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #2E86C1;
        --secondary: #48C9B0;
        --background: #F4F6F7;
        --card: #FFFFFF;
        --text-primary: #1B2631;
        --text-secondary: #566573;
        --success: #27AE60;
        --warning: #F39C12;
        --error: #E74C3C;
        --border: rgba(27, 38, 49, 0.08);
        --shadow: 0 10px 28px rgba(27, 38, 49, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f4f8fb 45%, #eef5fb 100%);
        color: var(--text-primary);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .main .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .top-nav {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        flex-wrap: wrap;
        background: linear-gradient(90deg, #1f5f8a 0%, var(--primary) 60%, #3a9fd6 100%);
        border-radius: 14px;
        padding: 0.65rem 0.8rem;
        box-shadow: 0 8px 24px rgba(27, 38, 49, 0.15);
        margin: 0.2rem 0 1rem 0;
    }
    .nav-chip {
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #f8fbff;
        background: rgba(255, 255, 255, 0.08);
        font-size: 0.88rem;
        font-weight: 600;
    }
    .nav-chip.active {
        background: #ffffff;
        color: #1f5f8a;
        border-color: #ffffff;
    }
    .dashboard-hero {
        background: linear-gradient(135deg, var(--primary) 0%, #4aa3d8 55%, #dff3fb 100%);
        color: white;
        padding: 2rem 2.25rem;
        border-radius: 24px;
        box-shadow: 0 14px 40px rgba(46, 134, 193, 0.18);
        margin-bottom: 1.25rem;
    }
    .dashboard-hero h1, .dashboard-hero p {
        margin: 0;
    }
    .dashboard-subtext {
        opacity: 0.95;
        font-size: 0.98rem;
        margin-top: 0.6rem;
        max-width: 980px;
    }
    .metric-card, .info-card, .result-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: var(--shadow);
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .metric-card:hover, .info-card:hover, .result-card:hover {
        transform: translateY(-2px);
        transition: all 0.25s ease;
        box-shadow: 0 14px 34px rgba(27, 38, 49, 0.10);
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.1rem;
    }
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.92rem;
    }
    .stButton > button {
        border-radius: 12px;
        padding: 0.55rem 1.2rem;
        border: none;
        background: var(--primary);
        color: white;
        font-weight: 600;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        box-shadow: 0 8px 18px rgba(46, 134, 193, 0.22);
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        background: #2471A3;
        color: white;
        transform: translateY(-1px);
    }
    div[data-testid="stFileUploader"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.7rem;
        box-shadow: var(--shadow);
    }
    div[data-testid="stFileUploader"] button {
        background: var(--secondary) !important;
        color: white !important;
        border-radius: 12px !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background: #2fb49d !important;
    }
    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid rgba(86, 101, 115, 0.18);
        box-shadow: 0 10px 28px rgba(27, 38, 49, 0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        padding: 0.45rem 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary);
        border-color: rgba(46, 134, 193, 0.35);
        box-shadow: 0 6px 16px rgba(46, 134, 193, 0.14);
    }
    .stMarkdown, .stText, p, li {
        color: var(--text-primary);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #174d70 0%, #1f5f8a 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f8fbff !important;
    }
    .stRadio [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        margin-bottom: 0.4rem;
        padding: 0.4rem 0.6rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .section-caption {
        color: var(--text-secondary);
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    .result-pill {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(72, 201, 176, 0.15);
        color: #0f766e;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .stCaption, .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px;
    }
    .stSpinner > div {
        border-top-color: var(--primary) !important;
        border-right-color: var(--secondary) !important;
    }
</style>
"""

DISEASE_LABELS = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
DISEASE_DESCRIPTIONS = {
    'CNV': 'Choroidal neovascularization may indicate abnormal blood vessel growth and requires specialist review.',
    'DME': 'Diabetic macular edema is associated with retinal swelling and fluid accumulation.',
    'DRUSEN': 'Drusen are deposits associated with early age-related macular degeneration.',
    'NORMAL': 'No major retinal abnormality detected by the current model output.',
}

if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = 0


def reset_upload():
    st.session_state.uploaded_file_key += 1
    st.session_state.pop("last_result", None)


def render_hero():
    st.markdown(
        """
        <div class="dashboard-hero">
            <h1>Human Eye Disease Detection System</h1>
            <p class="dashboard-subtext">
                A professional medical AI dashboard for reviewing eye images, predicting disease classes,
                and visualizing the highlighted region with enhancement and localization overlays.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav(current_page):
    pages = ["Home", "Predict", "About", "Model Info", "Instructions"]
    chips = []
    for page in pages:
        css_class = "nav-chip active" if page == current_page else "nav-chip"
        chips.append(f"<span class='{css_class}'>{page}</span>")
    st.markdown(f"<div class='top-nav'>{''.join(chips)}</div>", unsafe_allow_html=True)


def render_metric_cards():
    col1, col2, col3 = st.columns(3)
    metrics = [
        ("4", "Disease classes"),
        ("OpenCV", "Image enhancement & localization"),
        ("Streamlit", "Interactive dashboard UI"),
    ]
    for column, (value, label) in zip([col1, col2, col3], metrics):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_info_cards():
    c1, c2, c3 = st.columns(3)
    cards = [
        ("Upload", "Drop or browse a JPG/PNG eye image from your local system."),
        ("Predict", "Run the existing ML model to identify the disease class."),
        ("Visualize", "See the original, enhanced, and highlighted output images."),
    ]
    for column, (title, text) in zip([c1, c2, c3], cards):
        with column:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="section-title">{title}</div>
                    <div class="section-caption">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_about_section():
    st.markdown("## About")
    st.write(
        "This dashboard helps clinicians and students review OCT images using the existing trained model. "
        "It adds a modern interface, image preview, confidence output, and visual overlays without changing the ML core."
    )


def render_model_info_section():
    st.markdown("## Model Info")
    st.markdown(
        f"""
        - Model file: `{MODEL_PATH.name}`
        - Input image size: `224 x 224`
        - Preprocessing: OpenCV-based enhancement + MobileNetV3 preprocessing
        - Output classes: CNV, DME, DRUSEN, NORMAL
        """
    )


def render_instructions_section():
    st.markdown("## Instructions")
    st.markdown(
        """
        1. Open the **Predict** page.
        2. Upload a JPG or PNG eye image.
        3. Preview the image before prediction.
        4. Click **Predict Disease**.
        5. Review the result, confidence score, and highlighted output.
        """
    )


def _resolve_model_path_for_loading(model_path: Path) -> str:
    """Return a loadable model path for Keras, handling mislabeled HDF5 files."""
    with model_path.open("rb") as f:
        header = f.read(4)

    # Legacy Keras HDF5 models start with this signature.
    if header == b"\x89HDF":
        tmp_h5 = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        tmp_h5.close()
        Path(tmp_h5.name).write_bytes(model_path.read_bytes())
        return tmp_h5.name

    return str(model_path)

@st.cache_resource()
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    loadable_path = _resolve_model_path_for_loading(MODEL_PATH)
    model = tf.keras.models.load_model(loadable_path, compile=False)
    return model

def enhance_image(image_bgr):
    # Contrast enhancement using CLAHE in LAB color space.
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Noise removal and sharpening.
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
    return sharpened


def preprocess_image(image_bgr, target_size=(224, 224)):
    enhanced_bgr = enhance_image(image_bgr)
    resized = cv2.resize(enhanced_bgr, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(rgb.astype(np.float32), axis=0)
    x = preprocess_input(x)
    return x, enhanced_bgr


def predict_disease(model, preprocessed_batch):
    predictions = model.predict(preprocessed_batch, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    return class_index, confidence, predictions


def highlight_disease_area(original_bgr, enhanced_bgr):
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted = original_bgr.copy()
    region_mask = np.zeros_like(gray)
    min_area = max(100, (gray.shape[0] * gray.shape[1]) // 500)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)

    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(highlighted, 0.65, heatmap, 0.35, 0)
    heatmap_overlay[region_mask == 0] = highlighted[region_mask == 0]

    return heatmap_overlay, edges, thresh


def show_results(original_bgr, enhanced_bgr, highlighted_bgr, disease_name, confidence):
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "predicted_result.jpg"
    cv2.imwrite(str(output_path), highlighted_bgr)

    st.success("Prediction completed successfully.")
    result_col, confidence_col = st.columns(2)
    with result_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="section-title">Predicted Disease</div>
                <div class="result-pill">{disease_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with confidence_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="section-title">Confidence Score</div>
                <div class="metric-value">{confidence * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#f7fbff")
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[1].imshow(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Enhanced Image")
    axes[2].imshow(cv2.cvtColor(highlighted_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Disease Highlighted")
    for ax in axes:
        ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(f"Output saved to: {output_path}")
    return output_path

# Sidebar
st.markdown(APP_THEME, unsafe_allow_html=True)
st.sidebar.title("Dashboard")
app_mode = st.sidebar.radio(
    "Select Page",
    ["Home", "Predict", "About", "Model Info", "Instructions"],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.caption("Medical AI dashboard for OCT image screening")

render_top_nav(app_mode)

#Main Page
if app_mode == "Home":
    render_hero()
    render_metric_cards()
    st.markdown("### Welcome")
    st.write(
        "Use the sidebar to move between the prediction workspace, project overview, model details, and instructions. "
        "The dashboard is optimized for quick image review and clear clinical presentation."
    )
    render_info_cards()

#About Project
elif app_mode == "Predict":
    st.markdown("<div class='section-title'>Prediction Workspace</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Upload an eye image, preview it, and run the existing model to get the disease result.</div>", unsafe_allow_html=True)

    upload_col, preview_col = st.columns([1, 1.15])
    with upload_col:
        st.markdown("#### Upload Image")
        test_image = st.file_uploader(
            "Choose an eye image",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.uploaded_file_key}",
            label_visibility="collapsed",
        )
        uploaded_bytes = test_image.getvalue() if test_image is not None else None
        c1, c2 = st.columns(2)
        with c1:
            predict_clicked = st.button("Predict Disease", use_container_width=True)
        with c2:
            reset_clicked = st.button("Reset", use_container_width=True, on_click=reset_upload)

        if test_image is None and predict_clicked:
            st.error("Please upload an eye image before predicting.")

    with preview_col:
        st.markdown("#### Image Preview")
        if uploaded_bytes is not None:
            st.image(uploaded_bytes, use_container_width=True)
        else:
            st.info("Your uploaded image preview will appear here before prediction.")

    if test_image is not None and predict_clicked:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(test_image.name).suffix) as tmp_file:
            tmp_file.write(uploaded_bytes)
            temp_file_path = tmp_file.name

        with st.spinner("Analyzing image and generating results..."):
            model = load_model()
            original_bgr = cv2.imread(temp_file_path)
            if original_bgr is None:
                st.error("Could not read the uploaded image using OpenCV.")
                st.stop()

            preprocessed_batch, enhanced_bgr = preprocess_image(original_bgr)
            result_index, confidence, _ = predict_disease(model, preprocessed_batch)
            highlighted_bgr, edge_image, threshold_image = highlight_disease_area(original_bgr, enhanced_bgr)

            predicted_name = DISEASE_LABELS[result_index]
            show_results(original_bgr, enhanced_bgr, highlighted_bgr, predicted_name, confidence)

            st.session_state.last_result = predicted_name

        st.success("Prediction completed successfully.")

        st.markdown("#### Visualization")
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        with viz_col1:
            st.image(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        with viz_col2:
            st.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), caption="Enhanced Image", use_container_width=True)
        with viz_col3:
            st.image(cv2.cvtColor(highlighted_bgr, cv2.COLOR_BGR2RGB), caption="Disease Highlighted Image", use_container_width=True)

        st.markdown("#### Result Panel")
        result_col1, result_col2 = st.columns([1, 1])
        with result_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="section-title">Predicted Disease Name</div>
                    <div class="result-pill">{predicted_name}</div>
                    <p style="margin-top: 0.75rem; color: #475569;">{DISEASE_DESCRIPTIONS[predicted_name]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with result_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="section-title">Confidence</div>
                    <div class="metric-value">{confidence * 100:.2f}%</div>
                    <p style="margin-top: 0.75rem; color: #475569;">Higher values indicate greater model confidence.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Localization Debug View"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(edge_image, caption="Edge Detection", clamp=True, use_container_width=True)
            with col2:
                st.image(threshold_image, caption="Threshold + Contours Basis", clamp=True, use_container_width=True)

        with st.expander("Model Output"):
            st.write(f"Predicted Disease: **{predicted_name}**")
            st.write(f"Confidence Score: **{confidence * 100:.2f}%**")
            st.write(f"Output saved to: **{OUTPUT_DIR / 'predicted_result.jpg'}**")

        # Recommendation
        with st.expander("Learn More"):
            if result_index == 0:
                st.write('''
                    OCT scan showing *CNV with subretinal fluid.*
                ''')
                st.image(uploaded_bytes)
                st.markdown(cnv)
            elif result_index == 1:
                st.write('''
                    OCT scan showing *DME with retinal thickening and intraretinal fluid.*
                ''')
                st.image(uploaded_bytes)
                st.markdown(dme)
            elif result_index == 2:
                st.write('''
                    OCT scan showing *drusen deposits in early AMD.*
                ''')
                st.image(uploaded_bytes)
                st.markdown(drusen)
            elif result_index == 3:
                st.write('''
                    OCT scan showing a *normal retina with preserved foveal contour.*
                ''')
                st.image(uploaded_bytes)
                st.markdown(normal)

elif app_mode == "About":
    render_hero()
    render_about_section()

elif app_mode == "Model Info":
    render_hero()
    render_model_info_section()

elif app_mode == "Instructions":
    render_hero()
    render_instructions_section()