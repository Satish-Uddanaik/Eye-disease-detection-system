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
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "predicted_result.jpg"
    cv2.imwrite(str(output_path), highlighted_bgr)

    st.success(f"Predicted Disease: {disease_name}")
    st.info(f"Confidence Score: {confidence * 100:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Identification"])

#Main Page
if(app_mode=="Home"):
    # image_path = "home_page.jpeg"
    # st.image(image_path,use_column_width=True)
    st.markdown("""
    ## **OCT Retinal Analysis Platform**

#### **Welcome to the Retinal OCT Analysis Platform**

**Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

##### **Why OCT Matters**
OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

---

#### **Key Features of the Platform**

- **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
- **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
- **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.

---

#### **Understanding Retinal Diseases through OCT**

1. **Choroidal Neovascularization (CNV)**
   - Neovascular membrane with subretinal fluid
   
2. **Diabetic Macular Edema (DME)**
   - Retinal thickening with intraretinal fluid
   
3. **Drusen (Early AMD)**
   - Presence of multiple drusen deposits

4. **Normal Retina**
   - Preserved foveal contour, absence of fluid or edema

---

#### **About the Dataset**

Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
- **Normal**
- **CNV**
- **DME**
- **Drusen**

Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification. The images were obtained from various renowned medical centers worldwide and span across a diverse patient population, ensuring comprehensive coverage of different retinal conditions.

---

#### **Get Started**

- **Upload OCT Images**: Begin by uploading your OCT scans for analysis.
- **Explore Results**: View categorized scans and detailed diagnostic insights.
- **Learn More**: Dive deeper into the different retinal diseases and how OCT helps diagnose them.

---

#### **Contact Us**

Have questions or need assistance? [Contact our support team](mailto:arzankhan994@gmail.com) for more information on how to use the platform or integrate it into your clinical practice.

    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. 
                Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.
                (A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). 
                (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). 
                (Middle right) Multiple drusen (arrowheads) present in early AMD. 
                (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

                ---

                #### Content
                The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). 
                There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

                Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

                Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

                Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing exper- tise for verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The first tier of graders consisted of undergraduate and medical students who had taken and passed an OCT interpretation course review. This first tier of graders conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. The dataset selection and stratification process is displayed in a CONSORT-style diagram in Figure 2B. To account for human error in grading, a validation subset of 993 scans was graded separately by two ophthalmologist graders, with disagreement in clinical labels arbitrated by a senior retinal specialist.

                """)

#Prediction Page
elif(app_mode=="Disease Identification"):
    st.header("Welcome to the Retinal OCT Analysis Platform")
    test_image = st.file_uploader("Upload your Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        # Save to a temporary file and get its path
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name
    #Predict button
    if(st.button("Predict")) and test_image is not None:
        with st.spinner("Please Wait.."):
            model = load_model()
            original_bgr = cv2.imread(temp_file_path)
            if original_bgr is None:
                st.error("Could not read the uploaded image using OpenCV.")
                st.stop()

            preprocessed_batch, enhanced_bgr = preprocess_image(original_bgr)
            result_index, confidence, _ = predict_disease(model, preprocessed_batch)
            highlighted_bgr, edge_image, threshold_image = highlight_disease_area(original_bgr, enhanced_bgr)

            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            predicted_name = class_name[result_index]
            show_results(original_bgr, enhanced_bgr, highlighted_bgr, predicted_name, confidence)

        with st.expander("Localization Debug View"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(edge_image, caption="Edge Detection", clamp=True)
            with col2:
                st.image(threshold_image, caption="Threshold + Contours Basis", clamp=True)

        #Recommendation
        with st.expander("Learn More"):
            #CNV
            if(result_index==0):
                st.write('''
                    OCT scan showing *CNV with subretinal fluid.*
                ''')
                st.image(test_image)
                st.markdown(cnv)
        
            #DME
            elif(result_index==1):
                st.write('''
                    OCT scan showing *DME with retinal thickening and intraretinal fluid.*
                ''')
                st.image(test_image)
                st.markdown(dme)

            #DRUSEN
            elif(result_index==2):
                st.write('''
                    OCT scan showing *drusen deposits in early AMD.*
                ''')
                st.image(test_image)
                st.markdown(drusen)
                
            #NORMAL
            elif(result_index==3):
                st.write('''
                    OCT scan showing a *normal retina with preserved foveal contour.*
                ''')
                st.image(test_image)
                st.markdown(normal)