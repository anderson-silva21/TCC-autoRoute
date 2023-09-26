from flask import Flask, request, jsonify
import os
import base64
import io
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def save_base64_image(base64data, filename):
    try:
        image_data = base64.b64decode(base64data)
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'PNG')
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False


@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    if 'image' in data:
        image_data = data['image']
        if save_base64_image(image_data, 'uploaded_image.png'):
            return jsonify({"message": "Image uploaded and saved successfully."}), 200
        else:
            return jsonify({"error": "Failed to save the image."}), 500
    else:
        return jsonify({"error": "Invalid JSON data. Make sure it contains 'image' field."}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
