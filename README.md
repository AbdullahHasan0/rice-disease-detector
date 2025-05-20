# 🌾 Rice Disease Detector

A deep learning-powered web app to detect **rice crop diseases** from leaf images using **ResNet18** and **Streamlit**.  

> 🚜 Helping farmers make quicker decisions with AI-powered plant diagnostics.

---

## 📌 Project Motivation

Rice is a staple food for more than half the world’s population. Diseases like **Bacterial Blight**, **Blast**, and **False Smut** can severely reduce yield if not detected early.  
This app aims to:
- Identify rice diseases from leaf images
- Suggest possible mitigation measures
- Provide a low-cost, AI-based alternative to expert consultation

---

## 🧠 Dataset

- 📂 **Source**: [Kaggle – Rice Crop Diseases](https://www.kaggle.com/datasets/thegoanpanda/rice-crop-diseases)
- 🔢 **Total Classes**:
  - Bacterial Blight Disease
  - Blast Disease
  - Brown Spot Disease
  - False Smut Disease
- 🖼️ ~50 high-resolution leaf images per class

---

## 🛠️ Tech Stack

| Component   | Tool                      |
|-------------|---------------------------|
| Language    | Python 3.10               |
| Deep Learning | PyTorch, Torchvision     |
| Image Processing | OpenCV                |
| Deployment  | Streamlit                 |
| Evaluation  | Scikit-learn, Matplotlib  |

---

## 🖼️ Preprocessing Strategy

Images are resized and normalized to match ResNet18’s expected input.  
**Steps:**
- Convert BGR → RGB (OpenCV)
- Resize to **224×224**
- Apply `torchvision.transforms`:
  - Random horizontal/vertical flip
  - Random rotation and color jitter (train set)
  - Normalization (ImageNet mean & std)

---

## 🧠 Model: ResNet18

- Used `resnet18(pretrained=True)` from `torchvision.models`
- Final fully connected layer modified to `nn.Linear(in_features=512, out_features=4)`
- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Scheduler: StepLR  

---

## 📊 Results

| Metric        | Score   |
|---------------|---------|
| **Test Accuracy**  | 90%     |
| **F1-Score (avg)** | 88%     |
| **Inference Time** | ~15 ms/image |

✅ **Most classes achieved >85% precision**  
📉 Minor misclassifications between *Blast* and *False Smut*

### 📌 Confusion Matrix

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Bacterial Blight | 0.86      | 1.00   | 0.92     | 6       |
| Blast            | 1.00      | 0.80   | 0.89     | 5       |
| Brown Spot       | 0.67      | 1.00   | 0.80     | 2       |
| False Smut       | 1.00      | 0.86   | 0.92     | 7       |
| **Accuracy**     |           |        | **0.90** | **20**  |

## 🖼️ Prediction Examples

| Image | True Label | Predicted | Correct? |
|-------|------------|-----------|----------|
| ✅   | False Smut | False Smut | ✅        |
| ❌   | Blast       | False Smut | ❌        |
| ✅   | Brown Spot  | Brown Spot | ✅        |

> Red titles for wrong predictions, green for correct (visualization included).

---

## 🚀 Deployment Plan

### ✅ Completed
- [x] Data cleaning & augmentation  
- [x] Train/test/validation split (80/10/10)  
- [x] ResNet18 fine-tuning  
- [x] Evaluation and metrics  
- [x] Visualize predictions with color-coded labels
- [x] Build Streamlit interface  
- [x] Allow image upload for real-time predictions   

### 🔜 Next Steps 
- [ ] Display disease description and prevention tips  

---

## 🧪 How to Run

### 1. Clone Repo
```bash
git clone https://github.com/yourusername/rice-disease-detector.git
cd rice-disease-detector
```

### 2. Make Virtual Envoirement 
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
``` bash
pip install -r requirements.txt
```
### 4. Start Application
``` bash
streamlit run app.py
```

## 👨‍💻 Author  
#### Syed Abdullah Hasan  
#### Aspiring AI Engineer | Karachi, Pakistan
