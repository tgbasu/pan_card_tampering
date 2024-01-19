from flask import Flask, render_template, request
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def calculate_ssim(original_path, tampered_path, uploaded_path):
    original = cv2.imread(original_path)
    tampered = cv2.imread(tampered_path)

    # Resize the images to a common size
    common_size = (original.shape[1], original.shape[0])
    tampered = cv2.resize(tampered, common_size)

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

    (score, _) = structural_similarity(original_gray, tampered_gray, full=True)
    return score


def process_images(original_path, tampered_path, uploaded_path):
    original = cv2.imread(original_path)
    tampered = cv2.imread(tampered_path)
    uploaded = cv2.imread(uploaded_path)

    # Resize the uploaded image to match the original image dimensions
    uploaded = cv2.resize(uploaded, (original.shape[1], original.shape[0]))

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
    uploaded_gray = cv2.cvtColor(uploaded, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(uploaded, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return Image.fromarray(original), Image.fromarray(uploaded), Image.fromarray(diff), Image.fromarray(thresh)


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'user_upload.png')
        file.save(filename)

        original_path = 'pan_card_tampering/image/original.png'
        tampered_path = 'pan_card_tampering/image/tampered.png'
        original_img, uploaded_img, diff_img, thresh_img = process_images(original_path, tampered_path, filename)

        ssim_score = calculate_ssim(original_path, tampered_path, filename)


        # Set the SSIM threshold for classification
        ssim_threshold = 0.75
        classification = "Original" if ssim_score >= ssim_threshold else "Tampered"

        # Convert images to base64 strings for display in HTML
        original_img_base64 = image_to_base64(original_img)
        uploaded_img_base64 = image_to_base64(uploaded_img)
        diff_img_base64 = image_to_base64(diff_img)
        thresh_img_base64 = image_to_base64(thresh_img)

        return render_template('result.html', ssim_score=ssim_score, classification=classification,
                               original_img=original_img_base64, uploaded_img=uploaded_img_base64,
                               diff_img=diff_img_base64, thresh_img=thresh_img_base64)

    return render_template('index.html')



