import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

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

def modify_nose(image, nose_region, scale_x, scale_y):
    if not nose_region:
        return image
    x, y, w, h = nose_region
    nose_region = image[y:y+h, x:x+w]
    if nose_region.size == 0:
        return image

    new_w, new_h = int(w * scale_x), int(h * scale_y)
    resized_nose = cv2.resize(nose_region, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    mask = np.ones(resized_nose.shape[:2], dtype=np.uint8) * 255
    center = (x + w // 2, y + h // 2)
    blended_image = cv2.seamlessClone(resized_nose, image, mask, center, cv2.NORMAL_CLONE)
    
    return blended_image

st.title("AI-Powered Nose Shape Modifier")
profile_option = st.radio("Choose profile to modify:", ["Front Profile", "Side Profile"])

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    profile_type = "front" if profile_option == "Front Profile" else "side"
    nose_region, marked_image = detect_nose_landmarks(image, profile_type)

    if nose_region:
        st.image(marked_image, caption="Detected Nose Landmarks", use_column_width=True)
        st.write("### Suggested Nose Shapes for your face")
        shape_option = st.selectbox("Choose a nose shape:",
                                    ["Original", "Slimmer", "Wider", "Sharpened Tip", "Rounded Tip", "Shorter", "Longer"])

        shape_modifiers = {
            "Original": (1.0, 1.0),      
            "Slimmer": (0.85, 1.05),      
            "Wider": (1.15, 1.0),         
            "Sharpened Tip": (1.0, 0.85), 
            "Rounded Tip": (1.0, 1.15),   
            "Shorter": (1.0, 0.85),      
            "Longer": (1.0, 1.2)         
        }

        scale_x, scale_y = shape_modifiers[shape_option]
        
        if st.button("Apply Suggested Nose Shape"):
            modified_image = modify_nose(image.copy(), nose_region, scale_x, scale_y)
            st.image(modified_image, caption=f"Modified Nose - {shape_option}", use_column_width=True)
        
        st.write("### Custom Adjustments")
        scale_x = st.slider("Adjust Nose Width", 0.5, 2.0, 1.0, step=0.1)
        scale_y = st.slider("Adjust Nose Height", 0.5, 2.0, 1.0, step=0.1)
        if st.button("Apply Custom Nose Modification"):
            result_image = modify_nose(image.copy(), nose_region, scale_x, scale_y)
            st.image(result_image, caption="Custom Modified Nose", use_column_width=True)
    else:
        st.error("No face/nose detected. Please upload a clear photo.")

# Add Copyright Notice at the Bottom
st.markdown("---")
st.markdown("© Sonali Kadam")
st.markdown("AI-Powered Nose Modifier 2025. All rights reserved.")
