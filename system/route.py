import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

img = cv2.imread('mapa_aereoDois.png', cv2.IMREAD_GRAYSCALE)

# dilatação dos obstáculos
kernel = np.ones((200, 200), np.uint8)
img_dilated = cv2.dilate(img, kernel, iterations=1)

# centro da base do eixo X
start_node = (img.shape[0] - 1, img.shape[1] // 2)

# destino
goal_node = (50, 800)


def heuristic(node, goal, img):  # distância euclidiana
    penalty = 1.0
    distance = np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)
    penalty_term = penalty * img[node[0], node[1]] / 255.0
    return distance + penalty_term


def is_valid(node, img):  # verifica limites do mapa e se não é um obstáculo
    if (
        0 <= node[0] < img.shape[0] and
        0 <= node[1] < img.shape[1] and
        img[node[0], node[1]] == 0
    ):
        return True
    return False


def astar(img, start, goal):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {node: float('inf') for node in np.ndindex(img.shape)}
    g_score[start] = 0
    f_score = {node: float('inf') for node in np.ndindex(img.shape)}
    f_score[start] = heuristic(start, goal, img)

    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in [(current[0] - 1, current[1]), (current[0] + 1, current[1]),
                         (current[0], current[1] - 1), (current[0], current[1] + 1)]:
            if is_valid(neighbor, img):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + \
                        heuristic(neighbor, goal, img)
                    open_set.put((f_score[neighbor], neighbor))

    return None


path = astar(img_dilated, start_node, goal_node)

# caminho no mapa aéreo
if path:
    path_points = np.array(path)

    plt.plot(path_points[:, 1], path_points[:, 0], 'r-', linewidth=2)

    n = 70
    selected_points = path_points[::n]

    plt.plot(selected_points[:, 1], selected_points[:, 0], 'go', markersize=3)

plt.imshow(img, cmap='gray')
plt.title('Mapa Aéreo com Rota Encontrada')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.show()
