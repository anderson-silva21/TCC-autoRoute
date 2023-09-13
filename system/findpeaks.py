import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem
img = cv2.imread('../mapMeioMetroEUmMetro.png', cv2.IMREAD_GRAYSCALE)

# Inicializa listas para os pontos de início e fim dos platôs
pontos_inicio_plato = []
pontos_fim_plato = []
obstaculos = []

# Verifica se a imagem foi carregada com sucesso
if img is None:
    print("Imagem não carregada.")
else:
    # Calcula a segunda derivada da linha central da imagem
    derivada_segunda = np.gradient(np.gradient(img[299, :]))

    # Define um limite para identificar inícios e fins de platôs
    limite_derivada_segunda = 3.5

    # Percorre a segunda derivada para identificar inícios e fins de platôs
    em_plato = False  # Indica se está em um platô
    for i in range(len(derivada_segunda)):
        if derivada_segunda[i] < -limite_derivada_segunda:
            if not em_plato:
                pontos_inicio_plato.append(i)
            em_plato = True
        elif derivada_segunda[i] > limite_derivada_segunda:
            if em_plato:
                pontos_fim_plato.append(i - 1)
            em_plato = False

    # Cria uma lista de obstáculos
    obstaculos = [(pontos_inicio_plato[i], pontos_fim_plato[i], np.mean(img[299, pontos_inicio_plato[i]:pontos_fim_plato[i] + 1]))
                  for i in range(len(pontos_inicio_plato))]

    # Define limites mínimos para altura e diferença entre início e fim de obstáculos
    limite_altura_minima = 126
    limite_diferenca_x = 10

    # Filtra os obstáculos significativos
    obstaculos_significativos = [
        (inicio, fim, altura) for inicio, fim, altura in obstaculos
        if altura >= limite_altura_minima and (fim - inicio) >= limite_diferenca_x
    ]

    # Define as dimensões do mapa aéreo (ajuste conforme necessário)
    largura_mapa = 640
    altura_mapa = 480

    # Cria uma matriz para representar o mapa aéreo
    mapa_aereo = np.zeros((altura_mapa, largura_mapa))

    # Preenche o mapa aéreo com os obstáculos significativos
    for obstaculo in obstaculos_significativos:
        inicio_x, fim_x, altura_y = obstaculo
        altura_y = int(altura_y)
        inicio_x = max(0, inicio_x)
        fim_x = min(largura_mapa - 1, fim_x)
        if altura_y >= altura_mapa:
            altura_y = altura_mapa - 1
        for x in range(inicio_x, fim_x + 1):
            mapa_aereo[altura_y, x] = 1

    # Plota os gráficos lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Plota a linha central, inícios e fins dos platôs
    axs[0].plot(img[299, :], label='Linha Central')
    axs[0].plot(pontos_inicio_plato, img[299, pontos_inicio_plato],
                'go', label='Início do Plato')
    axs[0].plot(pontos_fim_plato, img[299, pontos_fim_plato],
                'ro', label='Fim do Plato')
    axs[0].set_title('Linha Central e Pontos de Início/Fim dos Platôs')
    axs[0].legend()

    # Plota o mapa aéreo
    axs[1].imshow(mapa_aereo, cmap='gray')
    axs[1].set_title('Mapa Aéreo')
    axs[1].set_xlabel('Eixo X')
    axs[1].set_ylabel('Eixo Y')

    plt.show()

    # Saída dos pontos de início e fim dos platôs
    print("Pontos de Início dos Platôs:", pontos_inicio_plato)
    print("Pontos de Fim dos Platôs:", pontos_fim_plato)

    # Saída da lista de obstáculos
    print("Lista de Obstáculos:")
    for obstaculo in obstaculos:
        print(
            f"Início X: {obstaculo[0]}, Fim X: {obstaculo[1]}, Altura Y: {obstaculo[2]}")

    # Exibe a lista de obstáculos significativos
    print("Lista de Obstáculos Significativos:", obstaculos_significativos)

    # Define o nome do arquivo de saída
    nome_arquivo_saida = 'mapa_aereo.png'

    # Salva o mapa aéreo como uma imagem
    # Multiplica por 255 para converter para valores de 0 a 255
    cv2.imwrite(nome_arquivo_saida, mapa_aereo * 255)

    # Exibe uma mensagem indicando que o mapa aéreo foi salvo com sucesso
    print(f"Mapa aéreo salvo como '{nome_arquivo_saida}'")
