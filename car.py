import cv2
import numpy as np

INPUT_TAG = 'input'
OUTPUT_TAG = 'output'

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


class Car:
    def __init__(self, first_contour, frame, num):
        self.num = num
        self.first_contour = first_contour
        self.frame_created = frame
        self.current_contour = None
        self.center = None
        self.last_frame_detected = None
        self.rect = None
        self.area_tag = {}
        self.add_contour(first_contour, frame)

    def add_contour(self, contour, frame):
        self.current_contour = contour
        self.center = self._calc_center()
        self.rect = cv2.boundingRect(contour)
        self.last_frame_detected = frame

    def _calc_center(self):
        M = cv2.moments(self.current_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY

    def get_distance(self, x, y):
        return abs(self.center[0] - x), abs(self.center[1] - y)

    def get_distance_value(self, x, y):
        x, y = self.get_distance(x, y)
        return x + y

    def draw(self, img):
        cv2.circle(img, self.center, 3, BLUE, -1)
        x, y, w, h = self.rect
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


class CarsController:
    def __init__(self, inputs, outputs, cross_area, treashold=50, frames_stay=5, frames_to_forget=15, min_size=40, max_size=200):
        self.frame = 1
        self.cars = []
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

    def add_contour(self, contour):
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        size = abs(w) + abs(h)
        # area = cv2.contourArea(contour)
        if size < self.min_size:
            return
        if size > self.max_size:
            return

        if self.frame == 1:
            self.cars.append(Car(contour, self.frame, len(self.cars)))

        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        for car in self.cars:
            frames_non_active = self.frame - car.last_frame_detected
            if frames_non_active == 0:
                continue
            if frames_non_active > self.frames_to_forget:
                continue
            distance = car.get_distance_value(cX, cY)
            if distance < self.distance_treshold - frames_non_active:
                car.add_contour(contour, self.frame)
                return
        self.cars.append(Car(contour, self.frame, len(self.cars)))

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


