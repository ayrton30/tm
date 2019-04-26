from ciratefi import ciratefi

radii = [0, 1, 3, 5]
scales = [0.4, 0.6, 0.7, 0.8, 0.9, 1, 1.1]
scales1 = [0.9, 1, 1.1]
angles = range(0, 360, 30)

ciratefi("t1.jpg", "tmp1.jpg", radii, scales, angles, t1=245, t2=241, threshold=0.65)