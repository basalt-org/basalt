import sys
from time.time import now
from python.python import Python
from utils.static_tuple import StaticTuple

from yolov8 import YoloV8, get_constant_values_from_onnx_model

import basalt.nn as nn
from basalt import Tensor, TensorShape, dtype
from basalt.utils.tensor_creation_utils import to_tensor, to_numpy


fn cam(
    inout model: nn.Model,
    constants: List[Tensor[dtype]]
) raises:

    Python.add_to_path("./examples")
    var yolo_utils = Python.import_module("yolo_v8_utils")
    
    var cv2 = Python.import_module("cv2")
    var np = Python.import_module("numpy")
    var cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    var height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT).to_float64()
    var width = cap.get(cv2.CAP_PROP_FRAME_WIDTH).to_float64()
    var length = max(height, width)
    var pads = np.array([0, length - height, 0, length - width, 0, 0], dtype=np.int32).reshape(3, 2)

    var last_time = now()

    while True:
        var r = cap.read()

        if not r[0]:
            print("Error: Could not read frame")
            break

        var image = np.pad(r[1], pads, mode='constant', constant_values=0)
        var blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(640, 640), swapRB=True)
        
        var res = model.inference(to_tensor(blob), constants[0], constants[1], constants[2])

        yolo_utils.draw_bounding_box_yolo(r[1], to_numpy(res[0]))
        cv2.imshow(
            'Basalt',
            cv2.putText(
                r[1],
                "FPS: " + String(1e9 / (now() - last_time)),
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1, cv2.LINE_AA
            )
        )

        last_time = now()
        if int(cv2.waitKey(1) & 0xFF) == 27 or cv2.getWindowProperty('Basalt', cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            sys.exit()


fn main():

    alias yolov8_n = StaticTuple[Float64, 3](
        0.33, 0.25, 2
    ) # d (depth_multiplier), w (width_multiplier), r (ratio)

    alias graph = YoloV8(1, yolov8_n)
    var model = nn.Model[graph]()

    model.load_model_data("./examples/data/yolov8n.onnx")
    
    try:
        var constant_values = get_constant_values_from_onnx_model("./examples/data/yolov8n.onnx")
        
        cam(
            model,
            constant_values
        )

    except e:
        print("Error in cam() function")
        print(e)
