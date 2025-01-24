import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Detect nose landmarks for the front profile
def detect_front_nose_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        nose_landmarks = [
            (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
            for lm in landmarks.landmark[1:9]  # Example nose points for front
        ]
        return nose_landmarks

# Detect nose region for side profile (bounding box)
def detect_side_nose_region(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]

        # Example: Use landmarks for the side profile nose (typically points around the nose and tip)
        nose_points = [
            (int(landmarks.landmark[i].x * image.shape[1]), int(landmarks.landmark[i].y * image.shape[0]))
            for i in [1, 2, 98, 327]  # Change these indices based on the side profile nose
        ]
        
        # Compute bounding box based on nose points
        xs, ys = zip(*nose_points)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

# Draw the nose landmarks or region
def draw_nose(image, nose_data, is_side=False):
    image_with_nose = image.copy()
    if is_side:
        x, y, w, h = nose_data
        cv2.rectangle(image_with_nose, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        for point in nose_data:
            cv2.circle(image_with_nose, point, 3, (0, 255, 0), -1)
    return image_with_nose

# Modify the nose shape with expanded region
def modify_nose(image, nose_data, scale_x, scale_y, is_side=False, increase_factor=1.2):
    if is_side:
        # For side profile (bounding box format)
        x, y, w, h = nose_data
        # Increase the bounding box size
        x -= int(increase_factor * w * 0.1)  # Increase x by 10% of width
        y -= int(increase_factor * h * 0.1)  # Increase y by 10% of height
        w = int(w * increase_factor)  # Increase width by the factor
        h = int(h * increase_factor)  # Increase height by the factor
    else:
        # For front profile (landmarks to bounding box)
        x, y, w, h = cv2.boundingRect(np.array(nose_data, dtype=np.int32))
        # Increase the bounding box size
        x -= int(increase_factor * w * 0.1)  # Increase x by 10% of width
        y -= int(increase_factor * h * 0.1)  # Increase y by 10% of height
        w = int(w * increase_factor)  # Increase width by the factor
        h = int(h * increase_factor)  # Increase height by the factor

    # Extract the nose region
    nose_region = image[y:y+h, x:x+w]
    resized_nose = cv2.resize(nose_region, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

    # Handle potential mismatched dimensions
    resized_h, resized_w = resized_nose.shape[:2]
    new_h = min(h, resized_h)
    new_w = min(w, resized_w)

    # Adjust the region to fit within the original image boundaries
    start_y = max(0, y)
    start_x = max(0, x)
    end_y = min(start_y + new_h, image.shape[0])
    end_x = min(start_x + new_w, image.shape[1])

    # Copy the resized nose region back into the image
    modified_image = image.copy()
    modified_image[start_y:end_y, start_x:end_x] = resized_nose[:end_y-start_y, :end_x-start_x]
    return modified_image

# Streamlit Interface
st.title("Nose Shape Visualizer")
st.write("Upload your front and side profile photos to visualize nose shape changes instantly.")

profile_option = st.radio("Choose profile to modify:", ["Front Profile", "Side Profile"])

if profile_option == "Front Profile":
    uploaded_file = st.file_uploader("Upload your front profile photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Front Profile", use_column_width=True)

        st.write("Processing...")
        nose_landmarks = detect_front_nose_landmarks(image)

        if nose_landmarks:
            image_with_nose = draw_nose(image, nose_landmarks)
            st.image(image_with_nose, caption="Nose Shape (Detected)", use_column_width=True)

            scale_x = st.slider("Adjust Nose Width", 0.5, 2.0, 1.0, step=0.1)
            scale_y = st.slider("Adjust Nose Height", 0.5, 2.0, 1.0, step=0.1)

            if st.button("Visualize Front Nose Modification"):
                result_image = modify_nose(image, nose_landmarks, scale_x, scale_y, increase_factor=1.2)
                st.image(result_image, caption="Modified Front Profile", use_column_width=True)
        else:
            st.error("No face/nose detected. Please upload a clear front profile photo.")

elif profile_option == "Side Profile":
    uploaded_side_file = st.file_uploader("Upload your side profile photo", type=["jpg", "jpeg", "png"])
    if uploaded_side_file:
        side_image = np.array(Image.open(uploaded_side_file))
        st.image(side_image, caption="Uploaded Side Profile", use_column_width=True)

        st.write("Processing...")
        nose_region = detect_side_nose_region(side_image)

        if nose_region is not None:
            image_with_nose = draw_nose(side_image, nose_region, is_side=True)
            st.image(image_with_nose, caption="Nose Region (Detected)", use_column_width=True)

            scale_x = st.slider("Adjust Nose Width", 0.5, 2.0, 1.0, step=0.1)
            scale_y = st.slider("Adjust Nose Height", 0.5, 2.0, 1.0, step=0.1)

            if st.button("Visualize Side Nose Modification"):
                result_image = modify_nose(side_image, nose_region, scale_x, scale_y, increase_factor=1.2, is_side=True)
                st.image(result_image, caption="Modified Side Profile", use_column_width=True)
        else:
            st.error("No nose detected. Please upload a clear side profile photo.")
