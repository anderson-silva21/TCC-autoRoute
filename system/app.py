import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import torch
import os
import glob
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        frame = cv2.imread(image_path)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img).to('cpu')

        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

            output = prediction.cpu().numpy()
            output = (output - output.min()) / (output.max() - output.min())
            output = (255 * output).astype(np.uint8)

        output_filename = os.path.splitext(filename)[0] + "_map.png"
        output_path = os.path.join(
            app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, output)

        return jsonify({'message': f'Saved {output_path}'})
