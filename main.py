import json

import cv2
import numpy as np

from car import CarsController
from collector import PolygonCollector


# Создаем объекты чтения готово видео и записи нового видео
from detectors.default import default_detector

vidcap = cv2. VideoCapture('example_3_1.mp4')
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Читаем первый кадр
success, last_image = vidcap.read()
import matplotlib.pyplot as plt

plt.imshow(last_image)
plt.show()


# Использовать сохраненный дамп размеченных зон
use_dump = True
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

frame_interval = 2


def process_default(frame_number, contours, returned_frame):
    for _ in range(frame_interval-1):
        cars_controller.increment_frame()
    cars_controller.add_contours(contours)
    copy = returned_frame.copy()
    cars_controller.draw(copy)
    out.write(copy)


default_detector(vidcap, process_default, frame_interval)

# закрываем каналы чтения и записи видео
out.release()
vidcap.release()
