
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


# # Custom CSS for UI enhancements
st.markdown("""
    <style>
        .stButton > button { 
            width: 100%; 
            border-radius: 8px; 
            background: linear-gradient(135deg, #ff416c, #ff4b2b); 
            color: white; 
            font-size: 16px; 
            padding: 10px; 
            border: none; 
            font-weight: bold;
            transition: all 0.3s ease-in-out; 
            box-shadow: 0px 4px 10px rgba(255, 65, 108, 0.4);
        }
        
""", unsafe_allow_html=True)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

def detect_nose_landmarks(image, profile="front", increase_factor=1.3):
    img_copy = image.copy()
    h, w, _ = image.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0]

        if profile == "front":
            nose_points = [4, 5, 195, 6, 19, 94, 97, 2]
        else:
            nose_points = [1, 2, 98, 327, 168, 122, 50]

        nose_coords = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in nose_points]

        for (x, y) in nose_coords:
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)

        xs, ys = zip(*nose_coords)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        nose_w, nose_h = x_max - x_min, y_max - y_min

        x_min = max(0, x_min - int(nose_w * (increase_factor - 1) / 2))
        y_min = max(0, y_min - int(nose_h * (increase_factor - 1) / 2))
        nose_w = min(w - x_min, int(nose_w * increase_factor))
        nose_h = min(h - y_min, int(nose_h * increase_factor))

        return (x_min, y_min, nose_w, nose_h), img_copy

def modify_nose(image, nose_region, scale_x=1.2, scale_y=1.2):
    if not nose_region:
        return image
    x, y, w, h = nose_region
    nose_roi = image[y:y+h, x:x+w].copy()
    if nose_roi.size == 0:
        return image

    new_w, new_h = int(w * scale_x), int(h * scale_y)
    resized_nose = cv2.resize(nose_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    x_center, y_center = x + w // 2, y + h // 2
    x1, y1 = max(0, x_center - new_w // 2), max(0, y_center - new_h // 2)
    x2, y2 = min(image.shape[1], x1 + new_w), min(image.shape[0], y1 + new_h)
    
    resized_nose = resized_nose[:y2-y1, :x2-x1]

    # Creating a mask for seamless cloning
    mask = 255 * np.ones(resized_nose.shape, resized_nose.dtype)

    # Blending using seamlessClone
    blended_image = image.copy()
    blended_image = cv2.seamlessClone(resized_nose, blended_image, mask, (x_center, y_center), cv2.NORMAL_CLONE)

    return blended_image

#Streamlit UI

st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: white;
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 10px 0;
        }
        .subtitle {
            font-size: 18px;
            text-align: center;
            color: #dddddd;
            font-style: italic;
            margin-bottom: 20px;
        }
    </style>
    
    <h1 class="title">👃 Rhinoplasty AI </h1>
    <p class="subtitle">✨ AI-powered tool to reshape and enhance your nose naturally! ✨</p>
""", unsafe_allow_html=True)


st.markdown("#### Modify your nose shape with AI technology in just a few clicks!")

# Choose between Camera or File Upload
input_choice = st.radio("Choose Image Input Method:", ["Upload Image", "Use Camera"])

# Initialize variables
img_source = None
profile_type = "front"  # Default for camera input

if input_choice == "Upload Image":
    profile_option = st.radio("📷 Upload your photo", ["Front Profile", "Side Profile"])
    profile_type = "front" if profile_option == "Front Profile" else "side"

    uploaded_file = st.file_uploader("Now, upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_source = uploaded_file

elif input_choice == "Use Camera":
    captured_file = st.camera_input("Now, capture an image 📷 ")
    if captured_file:
        img_source = captured_file  # Automatically assumes front profile

# Process the image if provided
if img_source:
    # Convert to NumPy array
    image = np.array(Image.open(img_source))

    nose_region, marked_image = detect_nose_landmarks(image, profile_type)

    if nose_region:
        st.image(marked_image, caption="✅ Detected Nose Landmarks", use_container_width=True)
        st.write("### 🎨 Suggested Nose Shapes for Your Face")


        shape_modifiers = {
            "Original": (1.0, 1.0),
            "Longer": (0.9, 1.1),
            "Wider": (1.2, 1.0),
            "Sharpened Tip": (0.95, 0.9),
            "Rounded Tip": (1.0, 1.2),
            "Shorter": (1.0, 0.85),
            "Slimmer": (0.9, 1.1),
        }

        shape_images = {shape: modify_nose(image.copy(), nose_region, scale_x, scale_y) for shape, (scale_x, scale_y) in shape_modifiers.items()}
        cols = st.columns(2)
        for i, (shape, img) in enumerate(shape_images.items()):
            with cols[i % 2]:  
                st.image(img, caption=f"🔹 {shape}", use_container_width=True)

        st.write("### 🎛️ Custom Adjustments")

        scale_x = st.slider("Adjust Nose Width", 0.5, 2.0, 1.0, step=0.1)
        scale_y = st.slider("Adjust Nose Height", 0.5, 2.0, 1.0, step=0.1)
        
        if st.button("Apply Custom Nose Modification"):
            modified_image = modify_nose(image.copy(), nose_region, scale_x, scale_y)

            # Display Original & Modified Images Side by Side
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="🔹 Original Image", use_container_width=True)
            with col2:
                st.image(modified_image, caption="✨ Modified Image", use_container_width=True)
    else:
        st.error("⚠️ No face/nose detected. Please upload a clear photo.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:14px;'>© 2025 Sonali Kadam | AI-Powered Nose Modifier</p>", unsafe_allow_html=True)



