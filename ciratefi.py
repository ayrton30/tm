import cv2
import numpy as np
import nms
from scipy.signal import convolve2d
import math
from tqdm import tqdm


def ciratefi(image, template, radii, scales, angles, t1=245, t2=245, t3=0.8, overlap_thresh=0.3):
    """
    finding a query template grayscale image 'template' in another grayscale image to analyze 'image', invariant to
    rotation, scale, translation, brightness and contrast

    :param image: path of the image to analyze
    :param template: path of the query template
    :param radii: set of radii
    :param scales: set of scales which the template is resize
    :param angles: set of rotation angles [°]
    :param t1: threshold for the first step Cifi,   range[0-255]
    :param t2: threshold for the second step Rafi,  range[0-255]
    :param t3: threshold for the third step Tefi,   range[0-1]
    :param overlap_thresh: for Non-Maximum Suppression algorithm
    :return: final pixel(s)
    """

    img = cv2.imread(image, 0)  # gray-scale image
    tmp = cv2.imread(template, 0)
    img_y, img_x = img.shape[:2]
    tmp_y, tmp_x = tmp.shape[:2]

# calculo de la matriz CA, the average grayscale of pixels of A(img) on the circle ring with radius rk centered at (x,y)
    kernel_ca = np.zeros((11, 11), np.float32)
    results = []

    for radius in radii:
        cv2.circle(kernel_ca, (5, 5), radius, 1, 1) # (5,5) es el centro del kernel de 11x11

        pk = np.sum(kernel_ca)
        conv = convolve2d(img, kernel_ca, mode='same', fillvalue=0)
        value = conv / pk
        results.append(value)

        kernel_ca.fill(0)
    ca = np.asarray(results)

# calculo de la matriz CT, the average grayscale of pixels of template(T) at scale si on the circle ring rk
# CT[i,k] | i=scales si | k=radii rk
    ct = np.zeros((len(scales), len(radii)), np.float32)

    for scale in scales:
        # para cada escala el template va a sufrir un cambio de tamaño
        tmp_x_scale = round(tmp_x * scale)
        tmp_y_scale = round(tmp_y * scale)

        tmp_resize = cv2.resize(tmp, None, fy=scale, fx=scale, interpolation=cv2.INTER_CUBIC)
        kernel_cq = np.zeros((tmp_y_scale, tmp_x_scale), np.float32)

        for radius in radii:
            # x0, y0 central pixel of T(template)
            x0 = round(tmp_x_scale / 2)
            y0 = round(tmp_y_scale / 2)

            cv2.circle(kernel_cq, (x0, y0), radius, 1, 1)

            pk = np.sum(kernel_cq)
            multi = np.multiply(tmp_resize, kernel_cq)
            value = np.sum(multi) / pk
            ct[scales.index(scale)][radii.index(radius)] = value

            kernel_cq.fill(0)

# calculo circular sampling correlation CisCorr
    cis_corr = np.zeros((img_y, img_x), np.float)   # circular sampling correlation CisCorr
    cis_ps = np.zeros((img_y, img_x), np.int)       # the probable scale of a pixel (x,y) -> best matching scale

    ct_norm = (ct - np.mean(ct)) / (np.std(ct) * len(ct))

    for y in range(0, img_y):
        for x in range(0, img_x):
            # vectores para normalizar y que los resultados de la correlacion varien [-1,1]
            ca_norm = (ca[:, y, x] - np.mean(ca[:, y, x])) / (np.std(ca[:, y, x]) * len(ca[:, y, x]))
            results = []

            for scale in scales:
                results.append(np.abs(np.correlate(ct_norm[scales.index(scale)], ca_norm, 'full')))
            results = np.asarray(results)

            cis_corr[y][x] = np.amax(results)
            # indice de la escala que hace maxima la corr
            cis_ps[y][x] = np.unravel_index(np.argmax(results, axis=None), results.shape)[0]
    # img resultado del primer filtro 'Cifi'
    cifi_img = np.zeros((img_y, img_x))
    cifi_img = cv2.normalize(cis_corr, cifi_img, 0, 255, cv2.NORM_MINMAX)
    cifi_img[cifi_img < t1] = 0

    first_pixels_y, first_pixels_x = np.nonzero(cifi_img >= t1)
    print("La cantidad de pixeles de primer grado son=", len(first_pixels_y))

    img_color = cv2.imread(image, cv2.IMREAD_COLOR)
    img_color[np.nonzero(cifi_img >= t1)] = [0, 33, 166]
    cv2.imwrite('first.jpg', img_color)

# calculo del vector RT, where T is radially sampled yielding a vector RT with m(angles) features
    length = max(radii)
    p1_tmp = (round(tmp_x / 2), round(tmp_y / 2)) #punto del centro del template

    kernel_rq = np.zeros((tmp_y, tmp_x), np.float32)
    rt = np.zeros(len(angles), np.float32)

    for angle in angles:
        theta = angle * math.pi / 180.0
        p2_tmp = (round(p1_tmp[0] + length * math.cos(theta)), round(p1_tmp[1] + length * math.sin(theta)))

        cv2.line(kernel_rq, p1_tmp, p2_tmp, 1, 1)
        multi = np.multiply(tmp, kernel_rq)
        value = np.sum(multi) / length

        rt[angles.index(angle)] = value
        kernel_rq.fill(0)

# LENTO!!!!
# Calculo de RA, the length of the radial lines is calculated according to the largest circle radius and the
# probable scale si computed by Cifi
    kernel_ra = np.zeros((img_y, img_x), np.float32)
    ra = np.zeros((img_y, img_x, len(angles)), np.float32)

    cos = []
    sin = []
    for angle in angles:
        cos.append(math.cos(angle * math.pi / 180.0))
        sin.append(math.sin(angle * math.pi / 180.0))

    # manipulando la imagen A
    for (x1, y1) in tqdm(zip(first_pixels_x, first_pixels_y), total=len(first_pixels_x)):
        for angle in (angles):  # progress bar
            index_ps = cis_ps[y1][x1]   # ps = possible scale
            length_ = scales[index_ps] * length

            # theta = angle * math.pi / 180.0
            x2 = round(x1 + length_ * cos[angles.index(angle)])
            y2 = round(y1 + length_ * sin[angles.index(angle)])

            cv2.line(kernel_ra, (x1, y1), (int(x2), int(y2)), 1, 1)
            multi = np.multiply(img, kernel_ra)
            value = np.sum(multi) / length_

            ra[y1][x1][angles.index(angle)] = value
            kernel_ra.fill(0)

# calculo radial sampling correlation RasCorr
    ras_corr = np.zeros((img_y, img_x), np.float)   # radial sampling correlation RasCorr at the best matching angle
    ras_ang = np.zeros((img_y, img_x), np.int)     # the probable scale of a pixel (x,y) -> best matching scale

    rt_norm = (rt - np.mean(rt)) / (np.std(rt) * len(rt))
    for (x, y) in zip(first_pixels_x, first_pixels_y):
        ra_norm = (ra[y, x] - np.mean(ra[y, x])) / (np.std(ra[y, x]) * len(ra[y, x]))

        results = []
        for angle in angles:
            cshift = np.roll(rt_norm, angles.index(angle))
            results.append(np.abs(np.correlate(cshift, ra_norm, 'full')))

        results = np.asarray(results)

        ras_corr[y][x] = np.amax(results)
        # indice del angulo que hace maxima la correlacion
        ras_ang[y][x] = np.unravel_index(np.argmax(results, axis=None), results.shape)[0]  #

    # img resultado del segundo filtro 'Rafi'
    rafi_img = np.zeros((img_y, img_x))
    rafi_img = cv2.normalize(ras_corr, rafi_img, 0, 255, cv2.NORM_MINMAX)
    rafi_img[rafi_img < t2] = 0

    second_pixels_y, second_pixels_x = np.nonzero(rafi_img >= t2)
    print("La cantidad de pixeles de segundo grado son=", len(second_pixels_y))

    img_color = cv2.imread(image, cv2.IMREAD_COLOR)
    for (x, y) in zip(second_pixels_x, second_pixels_y):
        top_left = x, y
        bottom_right = (top_left[0] + 2, top_left[1] + 2)
        cv2.rectangle(img_color, top_left, bottom_right, (120, 200, 100), 2)
        # print(x, y, scales[cis_ps[y][x]], angles[ras_ang[y][x]])
    cv2.imwrite('second.jpg', img_color)

# rotación y escala de la imagen, template matching
    final_pixelX = []
    final_pixelY = []

    # escalar el delta al si(scale_i) de mejor coincidencia
    delta_x = round(tmp_x / 2)  # para que no ocurra-> template > imagen (cropped)
    delta_y = round(tmp_y / 2)

    for (x, y) in zip(second_pixels_x, second_pixels_y):
        ang = angles[ras_ang[y][x]]
        sca = scales[cis_ps[y][x]]

        M = cv2.getRotationMatrix2D((x, y), ang, sca)
        affine = cv2.warpAffine(img.copy(), M, (img_x, img_y))

        a = y - delta_y
        b = x - delta_x

        if a < 0: a = 0
        if b < 0: b = 0

        cropped = affine[a:(a + tmp_y), b:(b + tmp_x)]
        res = cv2.matchTemplate(cropped, tmp, cv2.TM_CCORR_NORMED)
        # print("res:", np.mean(res))
        if np.mean(res) > t3:
            final_pixelX.append(x)
            final_pixelY.append(y)

# procesamiento de resultado, eleccion del mejor rectangulo
    boxes = []
    for (x, y) in zip(final_pixelX, final_pixelY):
        sca = scales[cis_ps[y][x]]
        delta_x_scale = round(delta_x * sca)    # cambio de escala para el recorte del template
        delta_y_scale = round(delta_y * sca)

        top_left = (x - delta_x_scale, y - delta_y_scale)
        bottom_right = (x + delta_x_scale, y + delta_y_scale)
        boxes.append([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
    boxes = np.asarray(boxes)
    pick = nms.non_max_suppression(boxes, overlap_thresh)
    print("La cantidad de pixeles finales son=", len(pick))

    img_color = cv2.imread(image, cv2.IMREAD_COLOR)
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(img_color, (startX, startY), (endX, endY), (120, 200, 100), 1)
    cv2.imwrite('final.jpg', img_color)

    return pick
