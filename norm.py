import cython

@cython.boundscheck(False)
cpdef unsigned char[:, :] cifi(int T, unsigned char [:, :] image):
    for y in range(0, img_y):
        for x in range(0, img_x):
        # vectores para normalizar y que los resultados de la correlacion varien [-1,1]
            ca_norm = (ca[:, y, x] - np.mean(ca[:, y, x])) / (np.std(ca[:, y, x]) * len(ca[:, y, x]))
            results = []

            for scale in scales:
                ct_norm = (ct[scales.index(scale)] - np.mean(ct[scales.index(scale)])) / (
                     np.std(ct[scales.index(scale)]) * len(ct[scales.index(scale)]))
                results.append(np.abs(np.correlate(ct_norm, ca_norm, 'full')))
            results = np.asarray(results)

            cis_corr[y][x] = np.amax(results)
        # indice de la escala que hace maxima la corr
            cis_ps[y][x] = np.unravel_index(np.argmax(results, axis=None), results.shape)[0]
