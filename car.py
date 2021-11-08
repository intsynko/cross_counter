import itertools
from typing import List

import cv2
import numpy as np

INPUT_TAG = 'input'
OUTPUT_TAG = 'output'

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


class Car:
    def __init__(self, rect, frame, num):
        self.num = num
        self.first_rect = rect
        self.frame_created = frame
        self.current_rect = None
        self.center = None
        self.last_frame_detected = None
        self.area_tag = {}
        self.add_rect(rect, frame)

    def add_rect(self, rect, frame):
        self.current_rect = rect
        self.center = self._calc_center()
        self.last_frame_detected = frame

    def _calc_center(self):
        x, y, w, h = self.current_rect
        cX = int(x + w / 2)
        cY = int(y + h / 2)
        return cX, cY

    def get_distance(self, x, y):
        return abs(self.center[0] - x), abs(self.center[1] - y)

    def get_distance_value(self, x, y):
        x, y = self.get_distance(x, y)
        return x + y

    def draw(self, img):
        cv2.circle(img, self.center, 3, BLUE, -1)
        x, y, w, h = self.current_rect
        cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(img, str(self.num), (x + 3, y), 1, 2, RED, 2)


class Area:
    def __init__(self, points, num, tag='input'):
        self.points = np.array(points, np.int32)
        self.num = num
        self.tag = tag
        # m = cv2.moments(points)
        # cx = int(m["m10"] / m["m00"])
        # cy = int(m["m01"] / m["m00"])
        # self.center = (cx, cy)
        self.counter = 0

    def calc_frame(self, cars):
        for car in cars:
            if cv2.pointPolygonTest(self.points, car.center, False) >= 0:
                self.counter += 1
                car.area_tag[self.tag] = self.num

    def draw(self, img):
        color = RED if self.tag == INPUT_TAG else BLUE
        cv2.polylines(img, np.int32([self.points]), isClosed=True, color=color)
        cv2.putText(img, str(self.counter), tuple(self.points[0]), 1, 2, RED, 2)
        cv2.putText(img, str(self.num), tuple(self.points[1]), 1, 2, BLUE, 2)


class CarsController:
    def __init__(self, inputs, outputs, cross_area, treashold=50, frames_stay=5, frames_to_forget=15, min_size=40, max_size=200):
        self.frame = 1
        self.cars: List[Car] = []
        self.inputs = [Area(input_, num, tag=INPUT_TAG) for num, input_ in enumerate(inputs)]
        self.outputs = [Area(output_, num, tag=OUTPUT_TAG) for num, output_ in enumerate(outputs)]
        self.cross_area = cross_area
        self.distance_treshold = treashold
        self.frames_stay = frames_stay
        self.frames_to_forget = frames_to_forget
        self.min_size = min_size
        self.max_size = max_size

    def get_last_frame_cars(self, tag):
        return [i for i in self.cars
                if i.last_frame_detected == self.frame - 1 and i.area_tag.get(tag) is None
                ]

    def add_contours(self, contours):
        for c in contours:
            self.add_contour(c)
        self.increment_frame()
        for input_ in self.inputs:
            input_.calc_frame(self.get_last_frame_cars(INPUT_TAG))
        for output_ in self.outputs:
            output_.calc_frame(self.get_last_frame_cars(OUTPUT_TAG))

    def add_rects(self, rects):
        for r in rects:
            self.add_rect(r)
        self.increment_frame()
        for input_ in self.inputs:
            input_.calc_frame(self.get_last_frame_cars(INPUT_TAG))
        for output_ in self.outputs:
            output_.calc_frame(self.get_last_frame_cars(OUTPUT_TAG))

    def add_rect(self, rect):
        x, y, w, h = rect
        size = abs(w) + abs(h)
        # area = cv2.contourArea(contour)
        if size < self.min_size:
            return
        if size > self.max_size:
            return

        if self.frame == 1:
            self.cars.append(Car(rect, self.frame, len(self.cars)))
        cX = x + w/2
        cY = y + h/2

        for car in self.cars:
            frames_non_active = self.frame - car.last_frame_detected
            if frames_non_active == 0:
                continue
            if frames_non_active > self.frames_to_forget:
                continue
            distance = car.get_distance_value(cX, cY)
            if distance < self.distance_treshold - frames_non_active:
                car.add_rect(rect, self.frame)
                return
        self.cars.append(Car(rect, self.frame, len(self.cars)))

    def add_contour(self, contour):
        rect = cv2.boundingRect(contour)
        self.add_rect(rect)

    def increment_frame(self):
        self.frame += 1

    def draw(self, img):
        for i in self.inputs:
            i.draw(img)
        for i in self.outputs:
            i.draw(img)

        for car in self.cars:
            if self.frame - car.last_frame_detected > self.frames_stay:
                continue
            car.draw(img)

    def str_results(self):
        msg = ""
        sort_cars = lambda cars_, tag: sorted(cars_, key=lambda car: car.area_tag.get(tag, -1))
        sorted_cars = sort_cars(self.cars, INPUT_TAG)
        for input_n, input_cars in itertools.groupby(sorted_cars, key=lambda car: car.area_tag.get(INPUT_TAG)):
            input_cars = list(input_cars)
            input_cars = sort_cars(input_cars, OUTPUT_TAG)
            if input_n is None:
                message = f"для {len(input_cars)} машин область старта не определена, "
            else:
                message = f"в области выезда №{input_n} проехало {len(input_cars)} машин, "
            msg += message
            for out_n, cars in itertools.groupby(input_cars, key=lambda car: car.area_tag.get(OUTPUT_TAG)):
                if input_n is None:
                    if out_n is None:
                        message = f"из них для {len(list(cars))} машин не опредлеена область съезда, "
                    else:
                        message = f"из них {len(list(cars))} съехало в {out_n} область, "
                elif out_n is None:
                    message = f"из них область съезда не определлена для {len(list(cars))} машин, "
                else:
                    message = f"из них в область съезда №{out_n} проехало {len(list(cars))} машин, "
                # message = f'{input_n}->{out_n} ({len(list(cars))})'
                msg += message
            msg += '\n'
        return msg
