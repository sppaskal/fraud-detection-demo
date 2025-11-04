import os
import fitz  # PyMuPDF - for reading PDF files and rendering pages as images
import cv2  # OpenCV - for image preprocessing (grayscale, thresholding)
import pytesseract  # Tesseract OCR
import numpy as np
from PIL import Image, ImageDraw  # Image handling and drawing overlays

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Directory where source PDFs/images will be mounted (from docker-compose)
source_dir = '/app/files'

# Directory to store annotated output images
output_dir = '/app/files/results'
os.makedirs(output_dir, exist_ok=True)

# PDF rendering
pdf_dpi = 200  # Resolution for rendering PDF pages

# OCR heuristics
confidence_threshold = 40  # Minimum confidence to trust OCR output
keyword_flags = ["sample", "template", "void"]  # Suspicious keywords
font_size_deviation_multiplier = 2  # Multiplier for std deviation threshold
fraud_score_max_trigger = 10  # Number of suspicious boxes to trigger max fraud score

# Overlay colors
color_low_confidence = "red"
color_size_deviation = "orange"


# ---------------------------------------------------------------------
# Convert PDF page to image
# ---------------------------------------------------------------------

def pdf_to_image(path):
    """Convert the first page of a PDF into an RGB numpy image."""
    doc = fitz.open(path)
    page = doc.load_page(0)  # load first page only for demo
    pix = page.get_pixmap(dpi=pdf_dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)


# ---------------------------------------------------------------------
# Preprocess image for OCR
# ---------------------------------------------------------------------

def preprocess(img):
    """Convert image to grayscale and apply binary thresholding for OCR clarity."""

    # If the image is already grayscale (2D), skip color conversion
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # thresholding for binarization (finds the optimal threshold to
    # separate foreground (text) from background)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# ---------------------------------------------------------------------
# Analyze OCR output for suspicious text or layout
# ---------------------------------------------------------------------

def analyze_text(img):
    """
    Run OCR on the image and flag suspicious regions:
    - Low confidence text detections
    - Suspicious keywords (e.g., 'sample', 'template', 'void')
    - Inconsistent font sizes (possible copy-paste edits)
    """
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    suspicious_boxes = []
    font_sizes = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue

        x, y, w, h = (
            data['left'][i],
            data['top'][i],
            data['width'][i],
            data['height'][i]
        )

        conf_value = data['conf'][i]
        # Handle confidence that may come as str, int, or float
        conf = int(conf_value) if isinstance(conf_value, (int, float)) else (
            int(conf_value) if str(conf_value).isdigit() else 0
        )

        # Collect height of all data points to check for variance later
        font_sizes.append(h)

        # Heuristic: flag low-confidence text or known template markers
        if conf < confidence_threshold or any(k in text.lower() for k in keyword_flags):
            suspicious_boxes.append(('low_confidence', (x, y, w, h)))

    # Heuristic: flag overall inconsistent font sizes (could suggest manual edits)
    mean = np.mean(font_sizes)
    std = np.std(font_sizes)
    threshold = font_size_deviation_multiplier * std
    for i, h in enumerate(font_sizes):
        if abs(h - mean) > threshold:
            suspicious_boxes.append(
                ('size_deviation', (
                    data['left'][i],
                    data['top'][i],
                    data['width'][i],
                    data['height'][i]
                ))
            )

    return suspicious_boxes


# ---------------------------------------------------------------------
# Draw overlays around suspicious regions
# ---------------------------------------------------------------------

def draw_overlay(img, boxes, out_path):
    """Draw colored rectangles around suspicious text regions based on suspicion type."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    for suspicion_type, (x, y, w, h) in boxes:
        color = "orange" if suspicion_type == "size_deviation" else "red"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

    pil_img.save(out_path)


# ---------------------------------------------------------------------
# Main loop - process all PDFs and images in mounted folder
# ---------------------------------------------------------------------

for name in os.listdir(source_dir):
    if name.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        path = os.path.join(source_dir, name)
        print(f"Processing: {name}")

        # Step 1: Convert PDF to image (if needed)
        if name.endswith('.pdf'):
            img = pdf_to_image(path)
        else:
            img = np.array(Image.open(path))

        # Step 2: Preprocess
        pre = preprocess(img)

        # Step 3: OCR and analysis
        boxes = analyze_text(pre)

        # Step 4: Visualization output
        out_file = os.path.join(
            output_dir, f"{os.path.splitext(name)[0]}_overlay.png"
        )
        draw_overlay(img, boxes, out_file)

        # Basic heuristic score: number of suspicious regions
        fraud_score = min(len(boxes) / fraud_score_max_trigger, 1.0)
        print(f"→ Fraud likelihood: {fraud_score:.2f}")
        print(f"→ Saved result: {out_file}\n")

print("All files processed.")
