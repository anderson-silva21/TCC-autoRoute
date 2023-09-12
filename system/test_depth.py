import cv2
import matplotlib.pyplot as plt

# Tente abrir e carregar a primeira imagem
img3 = cv2.imread('./mapMeioMetroEUmMetro.png')
if img3 is not None:
    plt.imshow(img3)
else:
    print("Não foi possível abrir a primeira imagem.")

# Plote 3 linhas da imagem (se a imagem foi carregada)
if img3 is not None:
    plt.figure()
    plt.plot(img3[249, :])
    plt.plot(img3[300, :])
    plt.plot(img3[350, :])
    plt.plot(img3[400, :])

# Tente abrir e carregar a segunda imagem
img4 = cv2.imread('./imgMeioMetroEUmMetro.png')
if img4 is not None:
    plt.figure()
    plt.imshow(img4)
else:
    print("Não foi possível abrir a segunda imagem.")

# Exiba os gráficos
plt.show()
