from flask import Flask, render_template, request, send_file
import os
import uuid
from rembg import remove
from PIL import Image
from io import BytesIO
import numpy as np
import face_recognition
from PIL import ImageEnhance

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# 📏 Size mapping (in pixels, assuming 300 DPI)
SIZE_MAP = {
    '1x1': (300, 300),
    '2x2': (600, 600),
    'passport': (413, 531),
}

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

        # Face auto-centering
        def auto_center_face(pil_image, target_size):
            np_image = np.array(pil_image.convert("RGB"))
            face_locations = face_recognition.face_locations(np_image)

            if not face_locations:
                # Fallback: just resize and center normally
                scale = min(target_size[0] / pil_image.width, target_size[1] / pil_image.height) * 0.75
                resized = pil_image.resize(
                    (int(pil_image.width * scale), int(pil_image.height * scale)),
                    Image.Resampling.LANCZOS
                )
                canvas = Image.new("RGBA", target_size, (255, 255, 255, 0))
                offset = (
                    (target_size[0] - resized.width) // 2,
                    (target_size[1] - resized.height) // 2
                )
                canvas.paste(resized, offset, resized)
                return canvas

            # Zoom out by increasing margin
            top, right, bottom, left = face_locations[0]
            face_margin = 2.0  # Zoom out more; try 1.8 to 2.2 if needed

            face_width = right - left
            face_height = bottom - top
            cx = (left + right) // 2
            cy = (top + bottom) // 2

            crop_w = int(face_width * face_margin)
            crop_h = int(face_height * face_margin)

            left_crop = max(cx - crop_w // 2, 0)
            top_crop = max(cy - int(crop_h * 0.7), 0)  # shift crop higher to show more top

            right_crop = min(cx + crop_w // 2, pil_image.width)
            bottom_crop = min(cy + crop_h // 2, pil_image.height)

            cropped = pil_image.crop((left_crop, top_crop, right_crop, bottom_crop))
            resized = cropped.resize(target_size, Image.Resampling.LANCZOS)

            # Shift face lower (positive y = lower face in frame)
            canvas = Image.new("RGBA", target_size, (255, 255, 255, 0))
            offset_y = int(0.01 * target_size[1])  # try 0.03 to 0.1 for small downward shift
            canvas.paste(resized, (0, offset_y), resized)

            return canvas


        target_size = SIZE_MAP.get(selected_size, (600, 600))
        centered = auto_center_face(image, target_size)

    
        if bg_option == 'white':
            # Merge with white background
            result = Image.new("RGB", target_size, (255, 255, 255))
            result.paste(centered, (0, 0), centered)
            processed_path = os.path.join(PROCESSED_FOLDER, filename.replace('.png', '.jpg'))
            result.save(processed_path, format="JPEG")
        else:
            # Transparent background
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
