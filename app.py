import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# -------------------- Streamlit UI Config --------------------
st.set_page_config(page_title="Rice Disease Detector", page_icon="🌾")
st.title("🌾 Rice Disease Detector")
st.markdown("""
Upload a clear image of a rice plant leaf and get instant disease classification.
""")

# -------------------- Load Model --------------------
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('models/rice_detector.pth', map_location=torch.device('cpu')))
model.eval()

# -------------------- Define Transform --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------- Image Upload --------------------
st.sidebar.header("🧪 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a rice leaf image...", type=["jpg", "jpeg", "png"])

# -------------------- Inference --------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    classes = ['Bacterial Blight Disease', 'Blast Disease', 'Brown Spot Disease', 'False Smut Disease']
    predicted_class = classes[pred_idx.item()]
    confidence_percent = confidence.item() * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: `{confidence_percent:.2f}%`")

# -------------------- 🔗 Footer --------------------
st.markdown("---")
st.markdown("Made with ❤️ by Syed Abdullah Hasan (https://github.com/AbdullahHasan0)")
