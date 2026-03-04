import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import driver

ENGINE_PATH = "yolox_s_fp16.engine"
IMAGE_PATH = "test.jpg"
INPUT_H, INPUT_W = 832, 832

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# -----------------------------
# Init CUDA (driver API)
# -----------------------------
driver.cuInit(0)

#If needed manual context creation is to be added here
# -----------------------------
# Load TensorRT engine
# -----------------------------
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read()) #CUDA context created internally, device is set o by default on jetpack
'''Deserialization is the process of loading a previously serialized TensorRT inference engine from a file (typically a .engine file) into memory for inference execution
This process involves:
  Reading the engine file into a memory buffer (e.g., a std::vector<uint8_t>). 
  Creating a TensorRT runtime object using nvinfer1::createInferRuntime. 
  Calling deserializeCudaEngine on the runtime to reconstruct the engine in memory from the serialized data
'''
#Engine loading
engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# -----------------------------
# Allocate buffers
# -----------------------------

host_buffers = {} #Dicts
device_buffers = {}
#TensorRT <= 8 had num_bindings. In TensorRT 10.x (JetPack 6.x) this API was removed
#TensorRT 10.x uses named I/O tensors instead of bindings
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    size = trt.volume(shape)

    host_mem = np.empty(size, dtype=dtype)
    _, device_mem = driver.cuMemAlloc(host_mem.nbytes)

    host_buffers[name] = host_mem
    device_buffers[name] = device_mem
    #print(f"{name}: shape={shape}, dtype={dtype}")

    #in TRT 10.x, binding order is not guaranteed, have to explicitly tel the execution context which tensor is input/output
    context.set_tensor_address(name, device_mem)
# -----------------------------
# Preprocess image
# -----------------------------
img = cv2.imread(IMAGE_PATH)
assert img is not None, "Image not found"
#Preprocessing
img = cv2.resize(img, (INPUT_W, INPUT_H))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0 #This makes the image RGB, fp32 and in range [0,1]. astype() is a NumPy method to change the data type of an array
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

host_buffers["images"][:] = img.ravel() #host_buffers id keyed by tensor name, input="images". There is no index 0

# -----------------------------
# H2D copy
# -----------------------------
#YOLOX input name is almost always "images"
driver.cuMemcpyHtoD(
    device_buffers["images"],
    host_buffers["images"].ctypes.data,
    host_buffers["images"].nbytes
)

# -----------------------------
# Inference
# -----------------------------
#stream 0 for default, can create a custom stream later for live feed and fps performance
context.execute_async_v3(0)
#Inference on GPU
# -----------------------------
# D2H copy
# -----------------------------
driver.cuMemcpyDtoH(
    host_buffers["output"].ctypes.data,
    device_buffers["output"],
    host_buffers["output"].nbytes
)

output = host_buffers["output"]
CLASS_NAMES = ["class0", "class1"]  # <-- replace with real names

detections = yolox_decode(
    output,
    img_h=832,
    img_w=832,
    num_classes=2,
    conf_thresh=0.3,
    nms_thresh=0.45
)
#OpenCV does not accept fp32 images in [0,1] of RGB order. It needs uint8 in [0,255] reange and BGR order.
#We need to do the conversions here
img_disp = (img.copy() * 255).astype(np.uint8)
img_disp = cv2.cvtcolor(img_disp, cv2.COLOR_RGB2BGR)
img_disp = draw_detections(img_disp, detections, CLASS_NAMES)

cv2.namedWindow("YOLOX TensorRT", cv2.WINDOW_NORMAL)
cv2.imshow("YOLOX TensorRT", img_disp) #OpenCV (GTK backend) is a little unstable on jetson when using Wayland display service. It is stable on X11
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Inference OK")
print("Output elements:", host_buffers["output"].size) 
#YOLOX uses 3 feature levels: P3 P4 and P5. The values vary with input size for inference. Based on the total value of these features, Output elements size here can help in verifying the number of classes in the custom value.
# output format per row: [cx, cy, w, h, obj_conf, cls0, cls1, ....]
print("First 10 values:", host_buffers["output"][:10]) #host_buffer is a dict, output tensor is accessed by name usually "output:

#Display of output in image window
#Output display is done with helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''Sigmoid as in maths function also called: logistic function or probability squashing function.
Neural networks often produce raw scores (logits) ranging from large negatives to large positives. But for inference (detection systems) we need values like probability of object, confidence score, binary classification output.
Sigmoid converts the logits(x) into these probabilities.
In YOLO detection heads, output is raw tensors not probabilities as seen in the output form, these values are the logits(too large) so sigmoid is applied to convert them.
For predicting box centers relative to grid sigmoid is used so as to keep the center offsets inside the grid cell (0-1 range)
Raw model predicts whether an object exists in the box, sigmoid converts that to probability of that object to exist in the box (0-1) giving the objectness score.
Similarly for predicting the class of the object, sigmoid gives the class probability and then confidence = objectness x class_prob'''

def nms(boxes, scores, iou_thresh=0.45):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

def yolox_decode(output, img_h, img_w, num_classes=2,
                 conf_thresh=0.3, nms_thresh=0.45):

    strides = [8, 16, 32]
    grids = []

    for stride in strides:
        h = img_h // stride
        w = img_w // stride
        yv, xv = np.meshgrid(np.arange(h), np.arange(w))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)

    grid = np.concatenate(grids, axis=0)
    output = output.reshape(-1, 5 + num_classes)

    # Decode boxes
    output[:, :2] = (output[:, :2] + grid) * np.repeat(strides, [10816, 2704, 676]).reshape(-1, 1)
    output[:, 2:4] = np.exp(output[:, 2:4]) * np.repeat(strides, [10816, 2704, 676]).reshape(-1, 1)

    # Convert to x1,y1,x2,y2
    boxes = np.zeros_like(output[:, :4])
    boxes[:, 0] = output[:, 0] - output[:, 2] / 2
    boxes[:, 1] = output[:, 1] - output[:, 3] / 2
    boxes[:, 2] = output[:, 0] + output[:, 2] / 2
    boxes[:, 3] = output[:, 1] + output[:, 3] / 2

    # Scores
    obj_conf = sigmoid(output[:, 4])
    class_scores = sigmoid(output[:, 5:])

    detections = []

    for cls in range(num_classes):
        scores = obj_conf * class_scores[:, cls]
        mask = scores > conf_thresh

        if not np.any(mask):
            continue

        cls_boxes = boxes[mask]
        cls_scores = scores[mask]

        keep = nms(cls_boxes, cls_scores, nms_thresh)

        for i in keep:
            detections.append([
                cls_boxes[i][0],
                cls_boxes[i][1],
                cls_boxes[i][2],
                cls_boxes[i][3],
                cls_scores[i],
                cls
            ])

    return detections

def draw_detections(img, detections, class_names):
    for det in detections:
        x1, y1, x2, y2, score, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        label = f"{class_names[cls]}: {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2
        )

    return img

driver.cuCtxDestroy(ctx)
