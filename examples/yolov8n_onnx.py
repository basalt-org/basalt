import cv2.dnn
import numpy as np
import onnxruntime as rt

from yolo_v8_utils import get_model_input, draw_bbox_from_image


def main(onnx_model, input_image):
    blob = get_model_input(input_image)
    
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    model = rt.InferenceSession(onnx_model)

    outputs = model.run(None, {"images": blob})[0]

    draw_bbox_from_image(input_image, outputs)


main(
    onnx_model='examples/data/yolov8n.onnx',
    input_image="examples/data/bus.jpg"
)