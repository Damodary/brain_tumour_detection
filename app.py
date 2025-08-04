import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from CNN_model import CNN

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# App Title and Instructions
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧠 Brain Tumor Detection App</h1>
    <p style='text-align: center;'>Upload an MRI scan image below to check for the presence of a brain tumor.</p>
    <hr style='border: 1px solid #ddd;'/>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📁 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess
    image = image.resize((128, 128))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.transpose((2, 0, 1))  # CHW format
    image = image / 255.0
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.round(output).item()
        confidence = float(output.item())

    # Output
    if prediction == 1:
        # Tumor Detected Message
        st.markdown("""
            <div style='background-color:#ffdddd;padding:20px;border-radius:10px;'>
                <h2 style='color:#d9534f;'>⚠️ Tumor Detected</h2>
                <p>Confidence: <b>{:.2f}</b></p>
            </div>
        """.format(confidence), unsafe_allow_html=True)

        # Symptoms
        st.markdown("""
            <h3 style='margin-top:20px;'>🩺 Common Symptoms of Brain Tumor</h3>
            <ul>
                <li>Persistent headaches</li>
                <li>Seizures or convulsions</li>
                <li>Blurred or double vision</li>
                <li>Nausea or vomiting</li>
                <li>Speech or hearing difficulties</li>
                <li>Changes in personality or behavior</li>
                <li>Weakness or numbness in limbs</li>
            </ul>
        """, unsafe_allow_html=True)

        # Treatments
        st.markdown("""
            <h3 style='margin-top:30px;'>🏥 Medical Treatments</h3>
            <ul>
                <li><b>Surgery</b>: Removal of the tumor (if accessible and safe)</li>
                <li><b>Radiation Therapy</b>: Targets and destroys tumor cells</li>
                <li><b>Chemotherapy</b>: Drug treatment to kill or shrink tumors</li>
                <li><b>Targeted Drug Therapy</b>: Focused treatment based on tumor type</li>
                <li><b>Immunotherapy</b>: Boosts body's immune system to fight tumor</li>
            </ul>
        """, unsafe_allow_html=True)

        # Remedies and Self-Care
        st.markdown("""
            <h3 style='margin-top:30px;'>🌿 Remedies & Lifestyle Support</h3>
            <ul>
                <li>Get enough sleep and rest</li>
                <li>Maintain a healthy diet rich in fruits and vegetables</li>
                <li>Manage stress through meditation and gentle yoga</li>
                <li>Avoid smoking and alcohol</li>
                <li>Regular checkups with neurologists or oncologists</li>
                <li>Join support groups for emotional health</li>
            </ul>
            <p style='color:gray;'>Note: Remedies are not a substitute for medical treatment. Always consult with a qualified healthcare provider.</p>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <div style='background-color:#ddffdd;padding:20px;border-radius:10px;'>
                <h2 style='color:#5cb85c;'>✅ No Tumor Detected</h2>
                <p>Confidence: <b>{:.2f}</b></p>
            </div>
        """.format(1 - confidence), unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style='border: 1px solid #ccc;'/>
    <p style='text-align: center; color: gray;'>© 2025 Brain Tumor Classifier | Built with 🧠 and ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
