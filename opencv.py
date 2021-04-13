import imutils
import cv2
import numpy as np
import pandas as pd


images_test = pd.read_pickle('../data/test.pkl')
idx = 0
for i in range(len(images_test)):
    img = images_test[i]
    img = np.repeat(img[..., np.newaxis], 3, -1)
    cv2.imwrite("a.jpg", img)
    image = cv2.imread("./a.jpg")

    # load the input image from disk, convert it to grayscale, and blur
    # it to reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blackAndWhiteImage = cv2.threshold(gray, 220, 255, cv2.THRESH_TOZERO)[1]
    blurred = cv2.GaussianBlur(blackAndWhiteImage, (3,3), 0)
    dilated = cv2.dilate(blackAndWhiteImage, None, iterations=1)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    img = []
    size = []

    # loop over the contours
    contour_number = 0
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # only iterate threw boxes that can be a digit  
        if (w >= 3 and w <= 35) and (h >= 3 and h <= 35):
            if(w*h > 50 and w*h < 500):
                roi = blackAndWhiteImage[(y):y + h, (x):x + w]
                img.append(roi)
                size.append(w*h)
                contour_number += 1

    top1 = 0
    top2 = 0
    top3 = 0
    idx1 = 0
    idx2 = 0
    idx3 = 0

    for i in range(len(size)):
        if i >= top1:
            top3 = top2
            top2 = top1
            top1 = size[i]
            idx3 = idx2
            idx2 = idx1
            idx1 = i

        elif i >= top2:
            top3 = top2
            top2 = size[i]
            idx3 = idx2
            idx2 = i

        elif i >= top3:
            top3 = size[i]
            idx3 = i


    nb_fig = len(size)

    if nb_fig == 3:
        for digit in img:
            cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), digit)
            idx += 1
    elif nb_fig > 3:
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[idx1])
        idx += 1
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[idx2])
        idx += 1
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[idx3])
        idx += 1
    elif nb_fig == 2:
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[0])
        idx += 1
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[1])
        idx += 1
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[1])
        idx += 1
    elif nb_fig == 1:
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[0])
        idx += 1
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[0])
        idx += 1
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), img[0])
        idx += 1
    else:
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), blackAndWhiteImage)
        idx += 1
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), blackAndWhiteImage)
        idx += 1
        cv2.imwrite('./images/0/DIGIT_{}.jpg'.format(idx), blackAndWhiteImage)
        idx += 1