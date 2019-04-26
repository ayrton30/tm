import cv2
import numpy as np
import nms
from scipy.signal import convolve2d
import math

# Necesito crear una matriz de 11x11xR, siendo R la cantidad de radios a evaluar

img = cv2.imread('mounts.png', 0)
img_y, img_x = img.shape[:2]

# number of circles (radii)
radii = [0, 1, 3, 5]

kernel = np.zeros((11, 11), np.float32)
#ca = np.zeros((img_y, img_x, len(radii)), np.float32)
test = []

#calculo de CA
for r in radii:
    kernel = cv2.circle(kernel, (5, 5), r, 1, 1)
    pk = np.sum(kernel)

    result = convolve2d(img, kernel, mode='same', fillvalue=0)
    valor = result / pk

    print(valor, valor.shape, "\n")
    test.append(valor)
    kernel.fill(0)

# 150*300*4 = 180000 pixles
ca = np.asarray(test)
print("ca.shape", ca.shape)
print(ca[:, 0, 0])


# Calculo de CQ- template

temp = cv2.imread('mount.png', 0)
tmp_y, tmp_x = temp.shape[:2]
# escalas
scales = [0.6, 0.7, 0.8, 0.9, 1, 1.1]

cq = np.zeros((len(scales), len(radii)), np.float32)

for scale in scales:
    new_x = round(tmp_x * scale)
    new_y = round(tmp_y * scale)
    #tmp_resize = np.zeros((new_y, new_x), np.uint8)
    tmp_resize = cv2.resize(temp, None, fy=scale, fx=scale, interpolation=cv2.INTER_CUBIC)

    kernel = np.zeros((new_y, new_x), np.float32)
    for radius in radii:
        # x0, y0 central pixel of Q (template)
        x0 = round(new_x / 2)
        y0 = round(new_y / 2)

        cv2.circle(kernel, (x0, y0), radius, 1, 1)
        pk = np.sum(kernel)

        result = np.multiply(tmp_resize, kernel)
        valor = np.sum(result) / pk

        cq[scales.index(scale)][radii.index(radius)] = valor
        kernel.fill(0)

print("cq\n", cq)

#Cálculo circular sampling correlation CisCorr

cis_corr = np.zeros((img_y, img_x), np.float)   # circular sampling correlation CisCorr
cis_ps = np.zeros((img_y, img_x), np.int)     # the probable scale of a pixel (x,y) -> best matching scale

#a = np.zeros((1, len(radii)), np.float32)
#b = np.zeros((1, len(radii)), np.float32)


for y in range(img_y):
    for x in range(img_x):

        b = (ca[:, y, x] - np.mean(ca[:, y, x])) / (np.std(ca[:, y, x]) * len(ca[:, y, x]))
        print(b)
        valores = []
        for scale in scales:
            a = (cq[scales.index(scale)] - np.mean(cq[scales.index(scale)])) / (np.std(cq[scales.index(scale)]) * len(cq[scales.index(scale)]))
            valores.append(np.abs(np.correlate(a, b, 'full')))

        valores = np.asarray(valores)

        cis_corr[y][x] = np.amax(valores)
        cis_ps[y][x] = np.unravel_index(np.argmax(valores, axis=None), valores.shape)[0]

print("cis_ps", cis_ps)

t1 = 245

first_img = np.zeros((img_y, img_x))
first_img = cv2.normalize(cis_corr, first_img, 0, 255, cv2.NORM_MINMAX)
first_img[first_img < t1] = 0


# First grade candidate pixels
# Posicion (y,x) en la imagen resultado del cifi de los pixeles
first_pixels_y, first_pixels_x = np.nonzero(first_img >= t1)
print("La cantidad de pares (x,y) pixeles de primer grado son=", len(first_pixels_y))

# pixeles_1 = zip(np.nonzero(first_img >= t1))

img_color = cv2.imread('mounts.png', cv2.IMREAD_COLOR)
img_color[np.nonzero(first_img >= t1)] = [0, 33, 166]
cv2.imwrite('first.jpg', img_color)


#
# Calculo RAFI

# Q is radially sampled using the largest sampling circle radius -> 5px
length = max(radii)

alpha = range(0, 360, 10)
p_centro = (round(tmp_x / 2), round(tmp_y / 2))

kernel_rafi_rq = np.zeros((tmp_y, tmp_x), np.float32)
rq = np.zeros(len(alpha), np.float32)

# calculo de RQ
for angle in alpha:
    θ = angle * math.pi / 180.0
    p2 = (round(p_centro[0] + length * math.cos(θ)), round(p_centro[1] + length * math.sin(θ)))

    cv2.line(kernel_rafi_rq, p_centro, p2, 1, 1)

    result = np.multiply(temp, kernel_rafi_rq)
    valor = np.sum(result) / length

    rq[alpha.index(angle)] = valor
    kernel_rafi_rq.fill(0)

# Calculo de RA
# The length of the radial lines is calculated according to the largest circle radius and the probable scale si computed
# by Cifi

# cis_ps[y][x] -> the probable scale
# First grade candidate pixels
# Posicion (y,x) en la imagen resultado del cifi de los pixeles

kernel_rafi_ra = np.zeros((img_y, img_x), np.float32)
ra = np.zeros((img_y, img_x, len(alpha)), np.float32)
# len(first_pixels_y) == len(first_pixels_x)

# for i, (pixelX, pixelY) in enumerate(zip(first_pixels_x, first_pixels_y)):
for angle in alpha:
    for (pixelX, pixelY) in zip(first_pixels_x, first_pixels_y):
        index_ps_scale = cis_ps[pixelY][pixelX]
        new_length = scales[index_ps_scale] * length

        θ = angle * math.pi / 180.0
        p2 = (round(pixelX + new_length * math.cos(θ)), round(pixelY + new_length * math.sin(θ)))
            
        cv2.line(kernel_rafi_ra, (pixelX, pixelY), (int(p2[0]), int(p2[1])), 1, 1)

        result = np.multiply(img, kernel_rafi_ra)
        valor = np.sum(result) / new_length

        ra[pixelY][pixelX][alpha.index(angle)] = valor
        kernel_rafi_ra.fill(0)

print("ra shape", ra.shape)

#Cálculo circular sampling correlation CisCorr

ras_corr = np.zeros((img_y, img_x), np.float)   # radial sampling correlation RasCorr at the best matching angle
ras_ang = np.zeros((img_y, img_x), np.int)     # the probable scale of a pixel (x,y) -> best matching scale

rq_normalizado = np.zeros((1, len(alpha)), np.float32)
ra_normalizado = np.zeros((1, len(alpha)), np.float32)

rq_normalizado = (rq - np.mean(rq)) / (np.std(rq) * len(rq))

for (pixelX, pixelY) in zip(first_pixels_x, first_pixels_y):
        ra_normalizado = (ra[pixelY, pixelX] - np.mean(ra[pixelY, pixelX])) / (np.std(ra[pixelY, pixelX]) * len(ra[pixelY, pixelX]))

        valores = []
        for angle in alpha:
            cshift =  np.roll(rq_normalizado, alpha.index(angle))
            valores.append(np.abs(np.correlate(cshift, ra_normalizado, 'full')))

        valores = np.asarray(valores)

        ras_corr[pixelY][pixelX] = np.amax(valores)
        ras_ang[pixelY][pixelX] = np.unravel_index(np.argmax(valores, axis=None), valores.shape)[0]

t2 = 245

second_img = np.zeros((img_y, img_x))
second_img = cv2.normalize(ras_corr, second_img, 0, 255, cv2.NORM_MINMAX)
second_img[second_img < t2] = 0


# First grade candidate pixels
# Posicion (y,x) en la imagen resultado del cifi de los pixeles
second_pixels_y, second_pixels_x = np.nonzero(second_img >= t2)
print("La cantidad de pares (x,y) pixeles de segundo grado son=", len(second_pixels_y))

# pixeles_1 = zip(np.nonzero(first_img >= t1))

img_color = cv2.imread('mounts.png', cv2.IMREAD_COLOR)
#img_color[np.nonzero(second_img >= t2)] = [0, 33, 166]

for (pixelX, pixelY) in zip(second_pixels_x, second_pixels_y):
    top_left = pixelX, pixelY
    bottom_right = (top_left[0] + 2, top_left[1] + 2)
    cv2.rectangle(img_color, top_left, bottom_right, (120, 200, 100), 2)
    print(pixelX, pixelY, scales[cis_ps[pixelY][pixelX]], alpha[ras_ang[pixelY][pixelX]])

cv2.imwrite('second.jpg', img_color)


#Rotacion y escala de la imagen, template matching

final_pixelX = []
final_pixelY = []

delta_x = round(tmp_x/2) + 3 # para que no ocurra-> template > imagen (cropped)
delta_y = round(tmp_y/2) + 3

for (pX, pY) in zip(second_pixels_x, second_pixels_y):
    angulo = ras_ang[pY][pX]
    escala = cis_ps[pY][pX]

    pX_escalado = round(pX * escala)
    pY_escalado = round(pY * escala)

    # cv2.getRotationMatrix2D(center, angle, scale) → retval
    # cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → ds
    M = cv2.getRotationMatrix2D((pX, pY), angulo, escala)
    affine = cv2.warpAffine(img.copy(), M, (img_x, img_y))

    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
    # dst = cv2.resize(affine, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    print("pY", pY, "pX", pX, "delta_y", delta_y, "delta_x", delta_x)
    a = pY - delta_y
    b = pX - delta_x

    if a < 0:
        a = 0
    if b < 0:
        b = 0
    cropped = affine[a:(a + tmp_y), b:(b + tmp_x)]
    print("cropped.shape", cropped.shape)
    print("temp.shape", temp.shape)
    res = cv2.matchTemplate(cropped, temp, cv2.TM_CCORR_NORMED)
    print("resultado", np.mean(res))

    if np.mean(res) > 0.81:
        final_pixelX.append(pX)
        final_pixelY.append(pY)

print(len(final_pixelX), len(final_pixelY))
img_color = cv2.imread('mounts.png', cv2.IMREAD_COLOR)

#Procesamiento de resultado, eleccion del mejor rectangulo
boxes = []
for (X, Y) in zip(final_pixelX, final_pixelY):
    top_left = X - delta_x, Y - delta_y
    bottom_right = (X + delta_x, Y + delta_x)
    boxes.append([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
    cv2.rectangle(img_color, top_left, bottom_right, (120, 200, 100), 1)

cv2.imwrite('final_1.jpg', img_color)
boxes = np.asarray(boxes)

# Resultados
pick = nms.non_max_suppression_slow(boxes, 0.3)
img_color = cv2.imread('mounts.png', cv2.IMREAD_COLOR)

for (startX, startY, endX, endY) in pick:
    # cv2.rectangle(img_color, top_left, bottom_right, (120, 200, 100), 1)
    cv2.rectangle(img_color, (startX, startY), (endX, endY), (120, 200, 100), 1)

cv2.imwrite('final_2.jpg', img_color)
