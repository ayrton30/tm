import cv2
import numpy as np
import numpy.ma as ma
import sys

np.set_printoptions(threshold=sys.maxsize)


img_template = cv2.imread('mount.png', 0)

height, width = img_template.shape
print("[img] ancho=", height, "\talto=", width)

# x0, y0 central pixel of Q (template)
x0 = round(width/2)
y0 = round(height/2)

# evaluo si hay numero par o impar de pixeles
if False:
    if x0 % 2 == 0:
        x0 = x0 - 1
    if y0 % 2 == 0:
        y0 = y0 - 1

rad_max = min(x0, y0)
print("rad_max=", rad_max)

l = 3 # number of circles + 1
n = [0.6, 0.7, 0.8, 0.9, 1, 1.1]


"""
coor => matriz que guarda las posiciones (x,y) de los pixeles de interes
filas       = cantidad de radios 'l'
columnas    = maximo de pixeles de interes -> Perimetro de la circunferencia maxima = 2*pi*radio
elemento    = posicion (x,y) dentro de la imagen

r1 | (px0, py0) |
r2 | (px0, py0) | (px1, py1)
.
.
.
rn | (px0, py0) | (px1, py1) | ... |

"""
"""
Cq[i,k] = sumatoria de pixeles de interes en radio determinado (par 'l') para diferentes escalas (n) de la imagen Q
"""

step = round(rad_max / l)
value_r = range(0, rad_max, step)
# voy a generar valores de radio desde 0px hasta el radio max, con un paso definido por la cantidad de radios que necesite

cq = ma.zeros((len(n), l), np.float)
radii = np.zeros((0,), dtype=int)

for scale in n:
    print("scale=", scale)
    circle_img = np.zeros((round(height * scale),
                           round(width * scale)), np.uint8) # circulo a partir del cual voy a obtener los pixeles de interes

    img_resize = np.zeros((round(height * scale),
                           round(width * scale)), np.uint8)

    #step = round(rad_max * scale / (l-1))
    #value_r = range(0, round(rad_max * scale), step)  # radios de pixeles

    img_resize = cv2.resize(img_template, dsize=(round(height * scale), round(width * scale)), interpolation=cv2.INTER_CUBIC)

    for radio in value_r:
        resultado, suma = -1, 0

        # In small scales, some of the outer circles may not fit inside the resized templates.
        # These circles are represented by a special value in table Cq (say, -1) and are not used to
        # compute the correlations.

        pk = round(2 * 3.14 * radio)
        if pk == 0:
            pk = 1

        radii = np.append(radii, radio)  # predefined circle radii

        cv2.circle(circle_img, (round(x0 * scale), (round(y0*scale))), radio, (255, 255, 255), 1)

        for y in range(0, circle_img.shape[0]):
            for x in range(0, circle_img.shape[1]):
                if circle_img[y, x] == 255:
                    suma = suma + img_resize[y, x]
                    resultado = suma/pk

        print("radio=", radio, "px")
        print("suma=", suma)
        print("resultado=", resultado)

        k = value_r.index(radio)    # 0 <= k < l 'set of predefined circle radii'
        i = n.index(scale)          # 0 <= i < n 'set of n scales'

        cq[i][k] = resultado
        if resultado == -1:
            cq[i][k] = ma.masked

        cv2.imwrite('t.jpg', circle_img)

        circle_img.fill(0)

print("\nradios", radii)
print("\nCq\n", cq)

#cv2.imshow('frog', frog)
#cv2.imwrite('t.jpg', circle_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()





# Cálculo de Ca[x,y,k] = CisA(x,y,rk)
# x = width | y = height
"""
         ______
        |      |
    y   |      |
        |______|
            x
"""
img = cv2.imread('mounts.png', 0)
height_img, width_img = img.shape
print("[img] ancho=", width_img, "\talto=", height_img)

ca = ma.zeros((width_img, height_img, len(value_r)), np.float)
circle_img = np.zeros((round(height_img * scale),
                       round(width_img * scale)), np.uint8) # circulo a partir del cual voy a obtener los pixeles de interes

for radio in value_r:
    resultado, suma = -1, 0

    # In small scales, some of the outer circles may not fit inside the resized templates.
    # These circles are represented by a special value in table Cq (say, -1) and are not used to
    # compute the correlations.

    pk = round(2 * 3.14 * radio)
    if pk == 0:
        pk = 1

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):

            cv2.circle(circle_img, (x, y), radio, (255, 255, 255), 1)

            if circle_img[y, x] == 255:
                suma = suma + img[y, x]
                resultado = suma / pk

            k = value_r.index(radio)  # 0 <= k < l 'set of predefined circle radii'
            ca[x][y][k] = resultado

            if resultado == -1:
                ca[x][y][k] = ma.masked

            circle_img.fill(0)

#print("\nCa[x=105][y=45][rk=0px]\n", ca)



#Cálculo circular sampling correlation CisCorr

cis_corr = np.zeros((width_img, height_img), np.float)

a = np.zeros((1, len(value_r)), dtype=np.float)
b = np.zeros((1, len(value_r)), dtype=np.float)


for scale in n:
    #Cq sin los valores de -1 para hacer la correlacion
    cq_compressed = cq[n.index(scale)].compressed()


    a = (cq[n.index(scale)] - np.mean(cq[n.index(scale)])) / (np.std(cq[n.index(scale)]) * len(cq[n.index(scale)]))
    print(a)

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):

            ca_compressed = ca[x][y].compressed()

            b = (ca[x][y] - np.mean(ca[x][y])) / (np.std(ca[x][y]) * len(ca[x][y]))
            cis_corr[x][y] = np.max(np.abs(np.correlate(a, b, 'full')))

print(cis_corr)
print("CisCorr.shape=", cis_corr.shape)
