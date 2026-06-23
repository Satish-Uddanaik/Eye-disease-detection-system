# 👁️ Eye Disease Detection System using Deep Learning

A deep learning-based web application for detecting retinal eye diseases from OCT (Optical Coherence Tomography) images. The system uses a Convolutional Neural Network (CNN) to classify eye images into different disease categories and provides visualizations along with disease-related recommendations.

## 🚀 Features

- Detects retinal diseases from OCT images
- Supports multiple disease classes:
  - CNV (Choroidal Neovascularization)
  - DME (Diabetic Macular Edema)
  - DRUSEN
  - NORMAL
- Image preprocessing:
  - Resizing
  - Noise Removal
  - Normalization
  - Enhancement using CLAHE
- Disease region highlighting
- Confidence score prediction
- Interactive Streamlit dashboard
- Disease information and recommendations

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend / AI
- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)

### Image Processing
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure

```text
Eye-disease-detection-system/
│
├── app.py
├── recommendation.py
├── Trained_Eye_disease_model_v2.keras
├── output/
│   └── predicted_result.jpg
│
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow

```text
Input OCT Image
        ↓
Image Preprocessing
(Resize, Noise Removal, Normalization)
        ↓
CNN Model
        ↓
Disease Prediction
        ↓
Confidence Score
        ↓
Disease Highlighting
        ↓
Recommendation & Precaution
```

---

## 📊 Model Performance

- Accuracy: **97% - 99%**
- High confidence predictions
- Automatic feature extraction using CNN
- Real-time prediction through web interface

---

## 📸 Screenshots

### Dashboard
- Home Page
- Prediction Workspace
- Image Preprocessing
- Disease Prediction
- Recommendation Section

(Add screenshots here)

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Eye-disease-detection-system.git
```

```bash
cd Eye-disease-detection-system
```

---

### 2️⃣ Create Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit tensorflow opencv-python matplotlib numpy
```

---

## ▶️ Run the Project

```bash
streamlit run app.py
```

After running, open:

```text
http://localhost:8501
```

---

## 🧪 How to Use

1. Open the application.
2. Navigate to the **Predict** page.
3. Upload an OCT image.
4. Click **Predict Disease**.
5. View:
   - Predicted Disease
   - Confidence Score
   - Highlighted Region
   - Recommendation

---

## 🎯 Advantages

- Early disease detection
- High accuracy
- Reduces manual effort
- Supports healthcare professionals
- Fast and user-friendly
- Useful for remote healthcare applications

---

## 🔮 Future Enhancements

- Support additional eye diseases
- Mobile application deployment
- Real-time camera integration
- Explainable AI visualizations
- Cloud deployment

---

## 👨‍💻 Authors

- Harsha M B
- Vedant S Varma
- Satish Halasiddappa Uddanaik

Department of Computer Science and Engineering  
University Visvesvaraya College of Engineering (UVCE)

---

## 📜 License

This project is developed for educational and research purposes.
