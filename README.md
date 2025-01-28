# Nose-Detector-and-Modifier-tool

This application allows users to upload front and side profile images to visualize and modify their nose shapes using computer vision techniques. The application uses the `mediapipe` library to detect facial landmarks, specifically for the nose area, and allows users to modify the width and height of the nose.

## Website deployed on : https://nose-detector-and-modifier-tool.onrender.com/

## Features
- **Front Profile**: Detects nose landmarks and allows users to modify the nose's width and height.
- **Side Profile**: Detects the nose region and allows users to modify its width and height.
- **Interactive Interface**: Users can upload images, see detected nose shapes, and adjust their appearance interactively.

## Technologies Used
- **Streamlit**: For building the web interface.
- **OpenCV**: For image manipulation.
- **MediaPipe**: For facial landmark detection.
- **Pillow (PIL)**: For image processing.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Sonali-2147/Nose Detector and Modifier tool.git
   cd Nose Detector and Modifier
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Open the URL shown in the terminal (usually `http://localhost:8501`) in your browser to start using the application.

## Usage

1. **Choose Profile**: Select whether you want to modify a "Front Profile" or a "Side Profile."
   
2. **Upload Image**: Click the "Upload" button to upload an image of your face. Make sure the face is clearly visible for the best results.

3. **Adjust Nose Dimensions**:
   - After the nose is detected, use the sliders to adjust the **Nose Width** and **Nose Height**.
   - Click "Visualize" to apply the changes and see the modified nose.

4. **Front Profile**: For the front profile, the system detects the nose landmarks around the face and allows the modification of the shape based on these points.

5. **Side Profile**: For the side profile, the system detects the nose region using bounding box coordinates and allows adjustments to the width and height.

6. **View Results**: After modifying, you will see the resulting image with the changes applied.

## Example Workflow

1. Upload your front or side profile photo.
2. The application detects the nose area and overlays it.
3. Adjust the sliders for nose width and height to visualize changes.
4. Click "Visualize" to see the modified image.

## Known Issues
- The application works best with clear and well-lit images.
- It may have difficulty detecting the nose in images where the face is obscured or turned at extreme angles.

## Contributing

Feel free to open an issue or a pull request if you want to contribute to this project!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Sreenshots
![Screenshot 2025-01-24 155719](https://github.com/user-attachments/assets/adac76de-0e7c-4cfa-b069-7452c0f70a14)

![Screenshot 2025-01-24 155726](https://github.com/user-attachments/assets/5c418405-5acb-45b0-b839-184861f21c38)

![Screenshot 2025-01-24 155756](https://github.com/user-attachments/assets/8781a35e-c60d-448c-adc6-332b1539851f)

![Screenshot 2025-01-24 155818](https://github.com/user-attachments/assets/5cbed567-b3a4-4157-bb23-36860cb3e333)


---

Let me know if you want to modify or add any additional details to the README!
