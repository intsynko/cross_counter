"""
 Дефолтный детектор контуров машин, основан на методах
"""
import cv2
import numpy as np


def default_detector(input, per_frame_function, frame_detection_interval=1):
    success, last_image = input.read()
    success, current_image = input.read()
    frame_num = 0
    while success:
        if frame_num % frame_detection_interval != 0:
            # per_frame_function(frame_num, [], current_image)
            continue

        # Переводим изображения в полутоновые
        grayA = cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

        # находим разницу соседних фреймов - это и будет показатель движущихся обеъктов
        diff_image = cv2.absdiff(grayB, grayA)

        # обрабатываем разницу пороговой обработкой
        ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

        # применяем дилатацию, эрозию
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        dilated = cv2.erode(dilated, kernel, iterations=1)

        # находим контуры
        contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        per_frame_function(frame_num, contours, current_image)

        last_image = current_image
        success, current_image = input.read()
