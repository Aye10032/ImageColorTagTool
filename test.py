from math import sqrt

from sklearn.cluster import KMeans
import cv2


def getSimilar(clusters):
    # 这里不用非得浮点数
    red = [231.00, 76.00, 60.00]
    purple = [201, 64, 192]
    blue = [52.00, 152.00, 219.00]
    green = [39.00, 174.00, 96.00]
    yellow = [241.00, 196.00, 15.00]
    orange = [230.00, 126.00, 34.00]
    white = [236.00, 240.00, 241.00]
    black = [44.00, 62.00, 80.00]
    colors = {
        'red': red,
        'purple': purple,
        'blue': blue,
        'green': green,
        'yellow': yellow,
        'orange': orange,
        'white': white,
        'black': black
    }
    for r1, g1, b1 in clusters.cluster_centers_:
        similar = 999
        color_result = ''
        for name, rgbs in colors.items():
            r2, g2, b2 = rgbs
            similar_new = sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b2 - b1) ** 2)
            if similar_new < similar:
                similar = similar_new
                color_result = name
        print(color_result)


src = cv2.imread('test.jpg')
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
dim = (500, 300)

img = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)

clt = KMeans(n_clusters=4)

dst = clt.fit(src.reshape(-1, 3))
getSimilar(dst)
