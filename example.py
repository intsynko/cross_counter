import json
import os

import cv2

from car import CarsController
from collector import PolygonCollector
from detectors import default_detector, web_detector


# Создаем объекты чтения готово видео и записи нового видео
vidcap = cv2. VideoCapture('pushkar_2.mp4')
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Читаем первый кадр
success, last_image = vidcap.read()


use_dump = True  # флаг использования сохраненного дампа размеченных зон
if not use_dump:
    # Просим разметить на первом кадре области
    inputs = PolygonCollector().collect(last_image, title='Разметьте области възда на перекресток (Закончив закройте окно)')
    outputs = PolygonCollector().collect(last_image, title='Разметьте области выъзда с перекрестка (Закончив закройте окно)')
    cross = PolygonCollector().collect(last_image, title='Разметьте перекресток', max_count=1)
    with open('dump.json', 'w') as f:
        f.write(json.dumps({'data': [inputs, outputs, cross]}))
else:
    # берем зоны из дампа
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
    max_size=400   # максимальный размер конутра
)

frame_interval = 2


def process_frame(frame_number, rects, returned_frame):
    for _ in range(frame_interval-1):
        cars_controller.increment_frame()
    cars_controller.add_rects(rects)
    copy = returned_frame.copy()
    cars_controller.draw(copy)
    out.write(copy)


# Можно использовать дефолтное распознование основаннное на методах цифровой обработки
# default_detector(vidcap, process_frame, frame_interval)

# Или использовать распознование с помощью нейросети
web_detector(vidcap, process_frame, frame_interval)

# закрываем каналы чтения и записи видео
out.release()
vidcap.release()

# пишем результаты
file = 'output.txt'
if os.path.isfile(file):
    os.remove(file)

with open(file, 'w') as f:
    results = cars_controller.str_results()
    print(results)
    f.write(results)
