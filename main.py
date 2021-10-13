import json

import cv2
import numpy as np

from car import CarsController
from collector import PolygonCollector


# Создаем объекты чтения готово видео и записи нового видео
vidcap = cv2. VideoCapture('example_3.mp4')
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Читаем первый кадр
success, last_image = vidcap.read()

# Использовать сохраненный дамп размеченных зон
use_dump = False
if not use_dump:
    # Просим разметить на первом кадре области
    inputs = PolygonCollector().collect(last_image, title='Разметьте области възда на перекресток (Закончив закройте окно)')
    outputs = PolygonCollector().collect(last_image, title='Разметьте области выъзда с перекрестка (Закончив закройте окно)')
    cross = PolygonCollector().collect(last_image, title='Разметьте перекресток', max_count=1)
    with open('dump.json', 'w') as f:
        f.write(json.dumps({'data': [inputs, outputs, cross]}))
else:
    with open('dump.json', 'r') as f:
        inputs, outputs, cross = json.loads(f.read())['data']


# Создаем объект-контроллер перекрестка
cars_controller = CarsController(
    inputs, outputs, cross,
    treashold=50,  # максимальное расстояние между центрами контуров одной на соседних кадрах
    frames_stay=2,  # кол-во фреймов без обнаружение машины, которое продолжать рисовать её контур в последнем месте
    frames_to_forget=15,  # кол-во фреймов без обнаружение машины,
    # чтобы перестать пытаться связывать машину с контурами со следующих кадров
    min_size=100,  # минимальный размер контура
    max_size=200   # максимальный размер конутра
)

# кол-во секунд для обработки
seconds = 120

for frame_num in range(0, seconds*10):
    success, current_image = vidcap.read()
    # пропускаем каждый второй кадр
    if frame_num % 2 != 0:
        img = current_image.copy()
        cars_controller.draw(img)
        out.write(img)
        last_image = current_image
        continue

    # Переводим изоюражения в полутоновые
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

    # грузим контуры в контроллер, он уже считает к какой машине какой контур принадлежит
    cars_controller.add_contours(contours)

    # рисуем на копии изображения машины и зоны
    dmy = last_image.copy()
    cars_controller.draw(dmy)
    # plt.imshow(dmy)
    # plt.show()
    out.write(dmy)

    last_image = current_image

# закрываем каналы чтения и записи видео
out.release()
vidcap.release()
