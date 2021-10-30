"""
 Детектор контуров машин, испольующий обученную нейросеть
"""
import os

from imageai.Detection import VideoObjectDetection


def web_detector(input, per_frame_function, frame_detection_interval=1, threshold=50, model_path=None):
    if model_path is None:
        model_path = os.path.join(os.getcwd(), "yolo.h5")

    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel("fast")

    def per_frame(frame_number, output_array, output_count, returned_frame=None):
        def convert(x1, y1, x2, y2):
            return x1, y1, x2 - x1, y2 - y1

        rects = [convert(*item['box_points']) for item in output_array
                 if item['name'] == 'car' and item['percentage_probability'] >= threshold]
        per_frame_function(frame_number, rects, returned_frame)

    detector.detectObjectsFromVideo(
        camera_input=input,
        save_detected_video=False,
        return_detected_frame=True,
        display_object_name=False,
        display_percentage_probability=False,
        display_box=False,
        per_frame_function=per_frame,
        log_progress=True,
        frame_detection_interval=frame_detection_interval,
    )
