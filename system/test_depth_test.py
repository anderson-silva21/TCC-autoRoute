import cv2
import matplotlib.pyplot as plt

# img3 = cv2.imread('../sequenciaTRES/maps/figuraUm_map.png')
# img3 = cv2.imread('../mapMeioMetroEUmMetro.png')

img3 = cv2.imread('../sequenciaTRES/maps/figuraUm_map.png')
if img3 is not None:
    plt.imshow(img3)
else:
    print("Não foi possível abrir a primeira imagem.")

if img3 is not None:
    plt.figure()
    plt.plot(img3[580, :])
    plt.plot(img3[680, :])


'''
# Tente abrir e carregar a segunda imagem
img4 = cv2.imread('../sequenciaTRES/figuraUm.jpeg')
if img4 is not None:
    plt.figure()
    plt.imshow(img4)
else:
    print("Não foi possível abrir a segunda imagem.")
'''

plt.show()
