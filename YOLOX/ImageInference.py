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

#Engine loading
engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# -----------------------------
# Allocate buffers
# -----------------------------

bindings = []
host_buffers = {}
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

    bindings.append(int(device_mem))
    host_buffers[name] = host_mem
    device_buffers[name] = device_mem

    print(f"{name}: shape={shape}, dtype={dtype}")

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
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

host_buffers[0][:] = img.ravel()

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
#stream 0 for default
context.execute_async_v3(0)

# -----------------------------
# D2H copy
# -----------------------------
driver.cuMemcpyDtoH(
    host_buffers["output"].ctypes.data,
    device_buffers["output"],
    host_buffers["output"].nbytes
)

print("Inference OK")
print("Output elements:", host_buffers[1].size)
print("First 10 values:", host_buffers[1][:10])

driver.cuCtxDestroy(ctx)
