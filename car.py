import cv2


class Car:
    def __init__(self, first_contour, frame, num):
        self.num = num
        self.first_contour = first_contour
        self.frame_created = frame
        self.current_contour = None
        self.center = None
        self.last_frame_detected = None
        self.rect = None
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


class CarsController:
    def __init__(self, inputs, outputs, cross_area, treashold=50, frames_stay=5, frames_to_forget=15, min_size=40, max_size=200):
        self.frame = 1
        self.cars = []
        self.inputs = inputs
        self.outputs = outputs
        self.cross_area = cross_area
        self.distance_treshold = treashold
        self.frames_stay = frames_stay
        self.frames_to_forget = frames_to_forget
        self.min_size = min_size
        self.max_size = max_size

    def add_contours(self, contours):
        for c in contours:
            self.add_contour(c)
        self.increment_frame()

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
        for car in self.cars:
            if self.frame - car.last_frame_detected > self.frames_stay:
                continue

            cv2.circle(img, car.center, 3, (255, 0, 0), -1)
            x, y, w, h = car.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(car.num),  (x + 3, y), 1, 2, (0, 0, 255), 2)


