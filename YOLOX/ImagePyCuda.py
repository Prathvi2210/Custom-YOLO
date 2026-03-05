import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "yolox_s_custom.engine"
IMAGE_PATH = "testImage.jpg"

INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.3
NMS_THRESH = 0.45

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def preprocess(img):
    h, w = img.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
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


def postprocess(outputs, scale, img_shape):
    boxes = [] #lists-simple, lightweight, ordered, mutable, duplicate values, indexed, dynamic sizes, nested but slow. So Images are stored as np arrays
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
    inputs, outputs, bindings = [], [], [] #In deep learning inference, buffers store: input imgs, intermediate tensors, model outputs
    #these buffers are later used to store the images when they are being used in GPU. They are 2D arrays, for ex: inputs first row will be the first input buffer which will have 2 elements of host and device.

    #stream is essentially a queue of GPU tasks, used to run GPU operations asynchronously(CPU can work while GPU is working) and in order(one after another)
    stream = cuda.Stream() #without streams CPU waits for GPU at every step
    
#In TRT bindings are the links between memory's I/O and the memory buffers that hold their data, they tell which buffer corresponds to each Input or Output when running inference
#TRT doesn't know where the data is stored, bindings provide the mapping. Each input and output tensor of the model has one binding.
#typical setup: bindings = [int(input_buffer), int(ouput_buffer)]. TRT now knows bindings[0] is the input memory
    for binding in engine:
        #info needed for allocating GPU memory
        size = trt.volume(engine.get_binding_shape(binding)) #how many elements(numbers) a tensor contains. the function in the bracet will return the values in the tensor and trt.volume will calculate the total i.e multiply the vaules in the tensor
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        #Allocation of memory buffers for TRT inference on CPU and GPU
        host_mem = cuda.pagelocked_empty(size, dtype) #NumPy array in piied RAM
        #Page-locked memory is fixed physical RAM location, Normal CPU memory can be moved around by OS/python
        #Pinned memory allows much faster transfers between CPU and GPU as it doesn't require a temporary buffer to locate the data. 
        device_mem = cuda.mem_alloc(host_mem.nbytes) #function inside bracket calculates how many bytes used by host buffer, so same can be allocated on GPu VRAM
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    img = cv2.imread(IMAGE_PATH)
    input_img, scale = preprocess(img)
    np.copyto(inputs[0][0], input_img.ravel()) #copy the preprocessed image data into the input buffer [line 81] that TRT will use for inference
#ravel() flattens a multidimensional array into a 1D array of the total number (multiplied) of elements. So basically, flatten image tensor in one dimension and add it to inputs buffer.
    # Inference
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream) #copy image tensor from host buffer [0],[0] to device buffer [0],[1], queued in the stream
    context.execute_async_v2(bindings, stream.handle) #Execution of neural network in GPU, also queued in stream
    for out in outputs:
        cuda.memcpy_dtoh_async(out[0], out[1], stream) #only copy the image tensors that are in outputs, queued in the stream
    stream.synchronize() #Wait until all tasks in the stream are finished

    output = outputs[0][0] #first output tensor in host buffer
    output = output.reshape(-1, output.shape[-1])

    detections = postprocess(output, scale, img.shape)

    for (box, score, cls) in detections:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"ID:{cls} {score:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display
    cv2.imshow("YOLOX TensorRT Inference", img)
    while True:
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() #Clean practice to use this function calling in python.
#When a python file runs, python sets __name__ = "__main__"
#When the file is imported as a module(import script) then __name__ = "script"
#so structuring the code like this in main() and calling ensures that the code runs only when the file is executed directly
