# importando dependencias
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# download MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# entrada pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Diretório de imagens
image_dir = "sequenciaTRES"
# Lista de arquivos no diretório
image_files = os.listdir(image_dir)

for image_file in image_files:
    # Ler a imagem do diretório
    image_path = os.path.join(image_dir, image_file)
    frame = cv2.imread(image_path)

    # Trasnformar entrada para o MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Fazer predição
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

        # Normalizar a saída para valores entre 0 e 255
        output = (output - output.min()) / (output.max() - output.min())
        output = (255 * output).astype(np.uint8)

    plt.imshow(output, cmap='gray')
    plt.pause(0.00001)

    # Salvar a imagem de saída
    output_filename = os.path.splitext(image_file)[0] + "_map.png"
    output_path = os.path.join(image_dir, output_filename)
    cv2.imwrite(output_path, output)
    print(f'Saved {output_path}')

plt.show()
