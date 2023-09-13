import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../sequenciaTRES/maps/figuraTres_map.png',
                 cv2.IMREAD_GRAYSCALE)

linha_central = 580

if img is None:
    print("Imagem não carregada.")
else:
    x = img[linha_central, :]

    # Aumente o tamanho da janela para a convolução
    window_size = 13  # Tamanho da janela, ímpar para manter o ponto central
    x = np.convolve(x, np.ones(window_size) / window_size, 'same')
    plt.plot(x)

    # segunda derivada da linha central da imagem
    dx = np.diff(x)
    plt.figure()
    plt.plot(dx)

    # picos positivos
    pks_positivos = []
    y_values_positivos = []
    while (np.max(dx) > 2.5) and len(pks_positivos) < 10:
        pos = np.argmax(dx)
        pks_positivos.append(pos)
        y_values_positivos.append(dx[pos])
        lo, hi = max(0, pos - 50), min(len(x), pos + 50)
        dx[lo:hi] = 0
    print("Picos Positivos:", pks_positivos)

    # picos negativos
    dx_negativo = -dx
    pks_negativos = []
    while (np.max(dx_negativo) > 2.5) and len(pks_negativos) < 10:
        pos = np.argmax(dx_negativo)
        pks_negativos.append(pos)
        lo, hi = max(0, pos - 50), min(len(x), pos + 50)
        dx_negativo[lo:hi] = 0
    print("Picos Negativos:", pks_negativos)

    # Crie a tupla de obstáculos
    obstaculos = list(
        zip(pks_positivos, pks_negativos, y_values_positivos))
    print("Obstáculos:", obstaculos)

    # Filtrar obstáculos com base nos critérios
    obstaculos_filtrados = [
        obstaculo for obstaculo in obstaculos if obstaculo[0] >= 5 or obstaculo[1] <= 1595]
    print("Obstáculos filtrados:", obstaculos_filtrados)

   # Define limites mínimos para altura e diferença entre início e fim de obstáculos
    limite_altura_minima = 126
    limite_diferenca_x = 10

    # Define as dimensões do mapa aéreo (ajuste conforme necessário)
    largura_mapa = 1600
    altura_mapa = 899

    # Cria uma matriz para representar o mapa aéreo
    mapa_aereo = np.zeros((altura_mapa, largura_mapa))

    # Preenche o mapa aéreo com os obstáculos significativos
    for obstaculo in obstaculos_filtrados:
        inicio_x, fim_x, altura_y = obstaculo
        altura_y = int(altura_y * 20)
        inicio_x = max(0, inicio_x)
        fim_x = min(largura_mapa - 1, fim_x)
        if altura_y >= altura_mapa:
            altura_y = altura_mapa - 1
        for x in range(inicio_x, fim_x + 1):
            mapa_aereo[altura_y, x] = 1

    plt.figure()
    plt.imshow(mapa_aereo, cmap='gray')
    plt.show()

    nome_arquivo_saida = 'mapa_aereoDois.png'
    cv2.imwrite(nome_arquivo_saida, mapa_aereo * 255)
    print(f"Mapa aéreo salvo como '{nome_arquivo_saida}'")
