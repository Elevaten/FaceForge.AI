from flask import Flask, render_template, request, send_file
import os
import uuid
from rembg import remove
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Size mapping (in pixels, assuming 300 DPI)
SIZE_MAP = {
    '1x1': (300, 300),
    '2x2': (600, 600),
    'passport': (413, 531),
}

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    selected_size = request.form.get('size', '2x2')
    bg_option = request.form.get('bg', 'white')  # 'white' or 'transparent'

    if file:
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Remove background using rembg
        with open(filepath, 'rb') as i:
            input_bytes = i.read()
            output_bytes = remove(input_bytes, alpha=(bg_option == 'transparent'))

        image = Image.open(BytesIO(output_bytes)).convert("RGBA")

        # Auto-center face using OpenCV
        def auto_center_face(pil_image, target_size):
            np_image = np.array(pil_image.convert("RGB"))
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                scale = min(target_size[0] / pil_image.width, target_size[1] / pil_image.height) * 0.75
                resized = pil_image.resize(
                    (int(pil_image.width * scale), int(pil_image.height * scale)),
                    Image.Resampling.LANCZOS
                )
                canvas = Image.new("RGBA", target_size, (255, 255, 255, 0))
                offset = (
                    (target_size[0] - resized.width) // 2,
                    (target_size[1] - resized.height) // 2 + int(0.05 * target_size[1])
                )
                canvas.paste(resized, offset, resized)
                return canvas

            x, y, w, h = faces[0]
            cx = x + w // 2
            cy = y + h // 2
            margin = 2.0

            crop_w = int(w * margin)
            crop_h = int(h * margin)

            left = max(cx - crop_w // 2, 0)
            top = max(cy - int(crop_h * 0.7), 0)
            right = min(cx + crop_w // 2, pil_image.width)
            bottom = min(cy + crop_h // 2, pil_image.height)

            cropped = pil_image.crop((left, top, right, bottom))
            resized = cropped.resize(target_size, Image.Resampling.LANCZOS)

            canvas = Image.new("RGBA", target_size, (255, 255, 255, 0))
            offset_y = int(0.01 * target_size[1])
            canvas.paste(resized, (0, offset_y), resized)

            return canvas

        target_size = SIZE_MAP.get(selected_size, (600, 600))
        centered = auto_center_face(image, target_size)

        if bg_option == 'white':
            result = Image.new("RGB", target_size, (255, 255, 255))
            result.paste(centered, (0, 0), centered)
            processed_path = os.path.join(PROCESSED_FOLDER, filename.replace('.png', '.jpg'))
            result.save(processed_path, format="JPEG")
        else:
            processed_path = os.path.join(PROCESSED_FOLDER, filename)
            centered.save(processed_path, format="PNG")

        return render_template('result.html', filename=os.path.basename(processed_path))

@app.route('/view/<filename>')
def view_file(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg' if filename.endswith('.jpg') else 'image/png')
    else:
        return "File not found", 404

@app.route('/processed/<filename>')
def download_file(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg' if filename.endswith('.jpg') else 'image/png',
                         as_attachment=True, download_name=filename)
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
