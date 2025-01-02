import streamlit as st
import os
import cv2
import pytesseract
import re
from PIL import Image
from io import BytesIO

# Configuration
UPLOAD_FOLDER = './static/uploads/'
REDACTED_FOLDER = './static/redacted/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REDACTED_FOLDER, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Regex patterns to detect sensitive information
SENSITIVE_PATTERNS = [
    r"\b\d{10}\b",                  # Phone numbers (10 digits)
    r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",  # PAN card numbers
    r"\b\d{2}/\d{2}/\d{4}\b",      # Dates (dd/mm/yyyy)
    r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", # Full names (simple pattern)
    r"\b\d{4}\s\d{4}\s\d{4}\b",  # Credit card numbers
    r"\b\d{1,3}\b",                # Ages (1 to 3 digits)
    r"Father'?s\sName[:\s]+[A-Za-z\s]+", # Father's Name
    r"S/o\s[A-Za-z\s]+"            # Father's Name (alternative)
]

# Function to redact sensitive content
def redact_sensitive_content(image_path, output_path, face_radius):
    redaction_level = 50  # Set redaction level to 50%

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate blur radius based on redaction level (scaled from 0-100 to a reasonable blur size)
    blur_radius = max(1, (redaction_level * 50) // 10 + 1)  # Scale blur with redaction level
    blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1

    face_blur_radius = max(1, (face_radius * 20) // 10 + 1)  # Scale face blur with face_radius
    face_blur_radius = face_blur_radius if face_blur_radius % 2 == 1 else face_blur_radius + 1

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        extended_y_start = max(0, y - int(h * 0.5))  # Extend upward to include hair
        extended_y_end = min(image.shape[0], y + int(h * 1.5))  # Extend downward to include shoulders
        extended_x_start = max(0, x - int(w * 0.2))  # Slightly extend width
        extended_x_end = min(image.shape[1], x + int(w * 1.2))

        face_region = image[extended_y_start:extended_y_end, extended_x_start:extended_x_end]
        face_region = cv2.GaussianBlur(face_region, (face_blur_radius, face_blur_radius), 0)
        image[extended_y_start:extended_y_end, extended_x_start:extended_x_end] = face_region

    # Detect text using OCR
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if any(re.search(pattern, text) for pattern in SENSITIVE_PATTERNS):
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            sensitive_region = image[y:y+h, x:x+w]
            blurred_region = cv2.GaussianBlur(sensitive_region, (blur_radius, blur_radius), 0)
            image[y:y+h, x:x+w] = blurred_region

    cv2.imwrite(output_path, image)

# Streamlit App
def main():
    st.title("Image Redaction Application")
    st.write("Upload an image to redact sensitive content!!")

    # User input for face blur radius
    face_radius = st.slider("Select level of redaction for face (0-100%)", 0, 100, 50)

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded file
        filename = uploaded_file.name.replace(" ", "_")  # Replace spaces with underscores
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        redacted_path = os.path.join(REDACTED_FOLDER, f"redacted_{filename}")

        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Perform redaction
        redact_sensitive_content(upload_path, redacted_path, face_radius)

        # Display original and redacted images
        st.image(Image.open(upload_path), caption="Original Image", use_column_width=True)
        st.image(Image.open(redacted_path), caption="Redacted Image", use_column_width=True)

        # Download redacted image
        with open(redacted_path, "rb") as file:
            btn = st.download_button(
                label="Download Redacted Image",
                data=file,
                file_name=f"redacted_{filename}",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
