
# 👃 Rhinoplasty AI - AI-Powered Nose Modifier

Rhinoplasty AI is a Streamlit-based web application that allows users to modify their nose shape using AI. The app detects nose landmarks using **MediaPipe FaceMesh** and provides real-time nose reshaping options.

---

## 🚀 Features

- 📷 Upload an image or capture one using the camera.
- 🔍 Detects nose landmarks using **MediaPipe FaceMesh**.
- 🎨 Provides multiple pre-defined nose shape modifications.
- 🎛️ Allows custom nose width and height adjustments.
- 🖼️ Displays original and modified images side by side.
- 🌟 Uses **seamless cloning** for natural blending.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/rhinoplasty-ai.git
cd rhinoplasty-ai
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App  
```bash
streamlit run app.py
```

---

## 📜 Requirements  

The required dependencies are listed in `requirements.txt`:

```txt
streamlit
opencv-python
mediapipe
numpy
pillow
```

Install them using:  
```bash
pip install -r requirements.txt
```

---

## 🎨 Usage

1. Choose an image input method:
   - **Upload an image** (Front or Side Profile).
   - **Use the camera** to capture an image.
2. The app detects nose landmarks.
3. Choose a **predefined nose shape** (e.g., Slimmer, Wider, Sharpened Tip, etc.).
4. Adjust nose **width and height** using sliders.
5. Click **"Apply Custom Nose Modification"** to see changes.

---

## 📸 Screenshots


![Screenshot 2025-02-26 140655](https://github.com/user-attachments/assets/40b208b5-f645-4830-908b-1a831b78ccd7)

![Screenshot 2025-02-26 140711](https://github.com/user-attachments/assets/890c1cae-90bc-41c1-b5f9-ebb8deba17b7)

![Screenshot 2025-02-26 140728](https://github.com/user-attachments/assets/2ae97111-ab29-46e2-ab86-81a6e5c4ceb7)



---

## 🤖 How It Works

1. **Face Detection**: Uses **MediaPipe FaceMesh** to extract nose landmarks.
2. **Region Extraction**: Extracts the detected nose region.
3. **Resizing & Transformation**: Resizes the nose based on user-selected shape.
4. **Seamless Cloning**: Blends the modified nose back into the face using **OpenCV's seamlessClone**.

---


## 👨‍💻 Author  

**Sonali Kadam**  
© 2025 AI-Powered Nose Modifier

---

## 📜 License  

This project is licensed under the MIT License.

---
```
