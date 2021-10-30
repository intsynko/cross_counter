import numpy as np
from imageai.Detection import VideoObjectDetection
import os
import cv2
import json

from car import CarsController
from collector import RectangleCollector

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel("fast")


vidcap = cv2.VideoCapture('example_3_1.mp4')
frame_width = int(vidcap.get(3))
frame_height = int(vidcap.get(4))
out = cv2.VideoWriter('output_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))


with open('dump.json', 'r') as f:
    inputs, outputs, cross = json.loads(f.read())['data']


# Создаем объект-контроллер перекрестка
cars_controller = CarsController(
    inputs, outputs, cross,
    treashold=50,  # максимальное расстояние между центрами контуров одной на соседних кадрах
    frames_stay=2,  # кол-во фреймов без обнаружение машины, которое продолжать рисовать её контур в последнем месте
    frames_to_forget=15,  # кол-во фреймов без обнаружение машины,
    # чтобы перестать пытаться связывать машину с контурами со следующих кадров
    min_size=50,  # минимальный размер контура
    max_size=200   # максимальный размер конутра
)


threashold = 50
frame_interval = 2


def per_frame(frame_number, output_array, output_count, returned_frame=None):
    def convert(x1, y1, x2, y2):
        return x1, y1, x2-x1, y2-y1
    countors = [convert(*item['box_points']) for item in output_array
                if item['name'] == 'car' and item['percentage_probability'] >= threashold]
    cars_controller.add_rects(countors)
    for _ in range(frame_interval-1):
        copy = returned_frame.copy()
        cars_controller.draw(copy)
        out.write(copy)


video_path = detector.detectObjectsFromVideo(
    camera_input=vidcap,
    # output_file_path=os.path.join(execution_path, "traffic_detected"),
    # frames_per_second=10,
    save_detected_video=False,
    return_detected_frame=True,
    display_object_name=False,
    display_percentage_probability=False,
    display_box=False,
    per_frame_function=per_frame,
    log_progress=True,
    frame_detection_interval=frame_interval,
)


# закрываем каналы чтения и записи видео
out.release()
vidcap.release()
