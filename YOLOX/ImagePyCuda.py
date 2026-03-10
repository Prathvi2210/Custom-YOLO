import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# Explicitly initialize CUDA, safer on jsetson than using pycuda.autoinit
cuda.init()
device = cuda.Device(0)
ctx = device.make_context()

ENGINE_PATH = "yolox_s_custom.engine"
IMAGE_PATH = "testImage.jpg"

CONF_THRESH = 0.3
NMS_THRESH = 0.45
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def get_input_hw(context, input_name): #This function is to automatically recognize what image size is needed by the engine.
    shape = context.get_tensor_shape(input_name)
    # shape = [1, 3, H, W]
    return shape[2], shape[3]
    
def preprocess(img):
    h, w = img.shape[:2]
    scale = min(input_w / w, input_h / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale))) #cv chooses INTER_LINEAR method for resizing by default. weighted avg of nearby pixels.
  #Other methods include INTER_NEAREST(picks nearest pixel), INTER_AREA(area-based avg) and INTER_CUBIC(uses more neighbours- higher quality)
    canvas = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8) #np.full is a function to create an array filled with a constant value.
  #114 here is the fill_value(pixel value) to fill the image. Every pixel is the (114, 114, 114)- Gray color, YOLO models often use 114 ass the padding color during preprocessing.
  #as YOLO used 114 padding during the training, we use it at inference as well
    canvas[:resized.shape[0], :resized.shape[1]] = resized
  #The resized image is copied onto this generated array image (canvas).
  #This is the letterbox padding step, done to ensure the input matches the fixed model input size.
  #Letterboxing preserves the aspect ratio which may get distorted during OpenCV resizing.
  #If model ratio and camera ratio is same and resize can preserve it, this step can be skipped and make it faster.

  #Convert an OpenCV image into tensor format for the neural network
    img = canvas.astype(np.float32) / 255.0 #Because Neural Nets operate using floating point numbers, then normalize
    img = img.transpose(2, 0, 1)  # OPenCV stores images in HWC, deep learning frameworks usually expect CHW
    img = np.expand_dims(img, axis=0) #Here we add a batch dimension, because neural networks process batches of images

    return img, scale #Final tensor format: (1, 3, H, W) float 32 0-1 range. The 1 is batch dimension-> 1 image, 3 is no. of channels

'''
def postprocess(outputs, scale, img_shape):
    boxes = [] #lists, they are simple, lightweight, ordered, mutable, duplicate values, indexed, dynamic sizes, nested but slow. So Images are stored as np arrays
# arrays are used instead when we need vectorized math. For ex: IoU comparison, NMS acceleration, box transformations
    scores = []
    class_ids = []

    for det in outputs:
        x, y, w, h, obj_conf = det[:5]
        class_conf = det[5:] #this will contain 2 values because my model has 2 classes
        cls_id = np.argmax(class_conf) #Returns index of the max value. Which class has highest probabilty of being that object
        conf = obj_conf * class_conf[cls_id] #effective confidence 

        if conf > CONF_THRESH:
            cx, cy = x / scale, y / scale
            bw, bh = w / scale, h / scale
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            boxes.append([x1, y1, int(bw), int(bh)])
            scores.append(float(conf))
            class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH) #inbuilt cv2 function for NMS
    return [(boxes[i], scores[i], class_ids[i]) for i in indices.flatten()] if len(indices) else [] #.flatten converts a multi dim array to single dimension
'''
def main():
  #with file ensures the file is closed after use and avoids memory leaks and resource constraints
    with open(ENGINE_PATH, "rb") as f: #rb is read binary, which is the format for TensorRT files (binary not text)
        runtime = trt.Runtime(TRT_LOGGER) #create a TensorRT runtine environment. The runtime loads engines, manages infernce execution and interacts with CUDA
      #TRT_LOGGER is a logging object, it controls how much TensorRT prints: INFO, WARNING, ERROR
        engine = runtime.deserialize_cuda_engine(f.read()) #load the engine in GPU executable form, convert the serialized engine file into an engine object
      #serialized means optimized model saved to disk, deserialization means reconstruct model in memory
      #The Engine object contains: optimized neural network, layer fusion, CUDA kernels, memory layout, TRT optimizations

    context = engine.create_execution_context() #runtime state of model execution. Engine is the factory blueprint, context is the running instance of the machine
    #the context is the object that holds runtime states, binds i/o memory and runs the model. It manages things that change per inference run: input tensor shapes, memory binds, execution states
    #One engine can have multiple contexts, this allows parallel inference. Good for video pipelines, multi-camera systems, high throughput servers
    #Basically it does the process: read input GPU memory > run the network > write output GPU memory

    # Allocate buffers(memory locations for temporarily storing input and output data while the model runs. They exist to mainly move data between CPU and GPU.
    #GPU cant directly read a numpy array like CPU. So we copy using buffers (memcpy) CPU is host and GPU is device
    host_buffers, device_buffers, bindings = [], [], [] #In deep learning inference, buffers store: input imgs, intermediate tensors, model outputs
    #these buffers are later used to store the images when they are being used in GPU. They are 2D arrays, for ex: inputs first row will be the first input buffer which will have 2 elements of host and device.

    #stream is essentially a queue of GPU tasks, used to run GPU operations asynchronously(CPU can work while GPU is working) and in order(one after another)
    stream = cuda.Stream() #without streams CPU waits for GPU at every step
    
#In TRT bindings are the links between memory's I/O and the memory buffers that hold their data, they tell which buffer corresponds to each Input or Output when running inference
#TRT doesn't know where the data is stored, bindings provide the mapping. Each input and output tensor of the model has one binding.
#typical setup: bindings = [int(input_buffer), int(ouput_buffer)]. TRT now knows bindings[0] is the input memory
    for i in range(engine.num_io_tensors): #engine.num_bindings are from TRT 8 and not used in 10
        #info needed for allocating GPU memory
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        size = trt.volume(shape) #how many elements(numbers) a tensor contains. the function in the bracet will return the values in the tensor and trt.volume will calculate the total i.e multiply the vaules in the tensor
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        #Allocation of memory buffers for TRT inference on CPU and GPU
        host_mem = cuda.pagelocked_empty(size, dtype) #NumPy array in piied RAM
        #Page-locked memory is fixed physical RAM location, Normal CPU memory can be moved around by OS/python
        #Pinned memory allows much faster transfers between CPU and GPU as it doesn't require a temporary buffer to locate the data. 
        device_mem = cuda.mem_alloc(host_mem.nbytes) #function inside bracket calculates how many bytes used by host buffer, so same can be allocated on GPu VRAM
        
        host_buffers[name] = host_mem
        device_buffers[name] = device_mem
        bindings.append(int(device_mem))

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    input_h, input_w = get_input_hw(context, input_name)

    img = cv2.imread(IMAGE_PATH)
    input_img, scale = preprocess(img, input_h, input_w)
    np.copyto(host_buffers[input_name], input_img.ravel()) #copy the preprocessed image data into the input buffer [line 81] that TRT will use for inference
#ravel() flattens a multidimensional array into a 1D array of the total number (multiplied) of elements. So basically, flatten image tensor in one dimension and add it to inputs buffer.
    # Inference
    cuda.memcpy_htod_async(device_buffers[input_name], host_buffers[input_name], stream) #copy image tensor from host buffer [0],[0] to device buffer [0],[1], queued in the stream
    #Execution of neural network in TensorRT GPU, also queued in stream
    context.execute_async_v3(stream_handle=stream.handle)) #TensorRT 10 removed execute_async_v2

    cuda.memcpy_dtoh_async(host_buffers[output_name], device_buffers[output_name], stream) #only copy the image tensors that are in outputs, queued in the stream
    stream.synchronize() #Wait until all tasks in the stream are finished

    #host_buffers[1] >>>first output tensor in host buffer
    # Expected output: [N, 6] -> x1, y1, x2, y2, score, class
    output = host_buffers[1].reshape(-1, 6) #reshape() changes the dimensions of a NumPy array without changing the data
    #output tensors shape is of the type (1, 8400, 85)=batch, detections, values per detection. 85 values represent [x, y, w, h, objectness, class 0 score, class 1 score, ....]. 
    #This is standard format because YOLO is trained on COCO which has 80 classes. [-1] tells numpy to automatically calculate this dimension, because custom models have variety of numbers of classes
    #So NumPy determines how many rows are needed to keep the total number of elements unchanged after reshaping. After reshaping the result is (8400, 85).
    #This is done because for post-processing it is easier to work with tensors without the batch dimension.
    #Basically this line flattens all dimensions except the last one so the tensor becomes a 2D array where each row represents one detection and each column represents prediction attributes.

    # detections = postprocess(output, scale, img.shape)
    '''
    for (box, score, cls) in detections:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"ID:{cls} {score:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    '''
    for det in output:
        x1, y1, x2, y2, score, cls = det
        if score < 0.3:
            continue

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )
        cv2.putText(
            img,
            f"ID:{int(cls)} {score:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # Display
    cv2.imshow("YOLOX TensorRT Inference", img)
    while True:
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()
    ctx.pop()

if __name__ == "__main__":
    main() #Clean practice to use this function calling in python.
#When a python file runs, python sets __name__ = "__main__"
#When the file is imported as a module(import script) then __name__ = "script"
#so structuring the code like this in main() and calling ensures that the code runs only when the file is executed directly
