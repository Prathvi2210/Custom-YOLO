import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Select engine
ENGINE_PATH = "yoloxs.engine"

IMAGE_PATH = "testImage.jpg"

TRT_LOGGER = trt.Logger(trt.Logger.INFO) #TRT uses a logging system to print messages to the console while running, the logger decides which messages should appear.
#This creates a logger with log level = INFO. It prints the INFO messages, Warning messages and error messages.

def preprocess(img, input_h, input_w):
    img_resized = cv2.resize(img, (input_w, input_h)) #cv chooses INTER_LINEAR method for resizing by default. weighted avg of nearby pixels.
  #Other methods include INTER_NEAREST(picks nearest pixel), INTER_AREA(area-based avg) and INTER_CUBIC(uses more neighbours- higher quality).
    #Convert an OpenCV image into tensor format for the neural network.
    img_resized = img_resized.astype(np.float32) # Because Neural Nets operate using floating point numbers.
    img_resized = img_resized.transpose(2, 0, 1) # OPenCV stores images in HWC, deep learning frameworks usually expect CHW.
    img_resized = np.expand_dims(img_resized, axis=0) #Here we add a batch dimension, because neural networks process batches of images.
    return np.ascontiguousarray(img_resized) #Final tensor format: (1, 3, H, W) float 32 0-1 range. The 1 is batch dimension-> 1 image, 3 is no. of channels.

def load_engine(engine_path):
    #with ensures the file is closed after use and avoids memory leaks and resource constraints.
    with open(engine_path, "rb") as f: #rb is read binary, which is the format for TensorRT files (binary not text).
        runtime = trt.Runtime(TRT_LOGGER) #create a TensorRT runtine environment. The runtime loads engines, manages infernce execution and interacts with CUDA.
      #TRT_LOGGER is a logging object, it controls how much TensorRT prints: INFO, WARNING, ERROR.
        engine = runtime.deserialize_cuda_engine(f.read()) #load the engine in GPU executable form, convert the serialized engine file into an engine object.
      #serialized means optimized model saved to disk, deserialization means reconstruct model in memory.
      #The Engine object contains: optimized neural network, layer fusion, CUDA kernels, memory layout, TRT optimizations.
    if engine is None:
        raise RuntimeError("Failed to load engine.")
    return engine


def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context() #runtime is state of model execution. Engine is the factory blueprint, context is the running instance of the machine.
    #the context is the object that holds runtime states, binds i/o memory and runs the model. It manages things that change per inference run: input tensor shapes, memory binds, execution states.
    #One engine can have multiple contexts, this allows parallel inference. Good for video pipelines, multi-camera systems, high throughput servers.
    #Basically it does the process: read input GPU memory > run the network > write output GPU memory.

    #In TRT bindings are the links between memory's I/O and the memory buffers that hold their data, they tell which buffer corresponds to each Input or Output when running inference
    #TRT doesn't know where the data is stored, bindings provide the mapping. Each input and output tensor of the model has one binding.
    #typical setup: bindings = [int(input_buffer), int(ouput_buffer)]. TRT now knows bindings[0] is the input memory
    for i in range(engine.num_io_tensors): #engine.num_bindings are from TRT 8 and not used in 10
        #info needed for allocating GPU memory
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    input_shape = engine.get_tensor_shape(input_name)
    context.set_input_shape(input_name, input_shape)

    batch, ch, input_h, input_w = input_shape

    #stream is essentially a queue of GPU tasks, used to run GPU operations asynchronously(CPU can work while GPU is working) and in order(one after another)
    stream = cuda.Stream() #without streams CPU waits for GPU at every step

    # Allocate buffers(memory locations for temporarily storing input and output data while the model runs. They exist to mainly move data between CPU and GPU.
    # GPU cant directly read a numpy array like CPU. So we copy using buffers (memcpy) CPU is host and GPU is device
    host_buffers = {}#In deep learning inference, buffers store: input imgs, intermediate tensors, model outputs
    #these buffers are later used to store the images when they are being used in GPU. 
    device_buffers = {} #They are 2D arrays, for ex: inputs first row will be the first input buffer which will have 2 elements of host and device.

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape) #how many elements(numbers) a tensor contains. the function in the bracet will return the values in the tensor and trt.volume will calculate the total i.e multiply the vaules in the tensor.
        
        #Allocation of memory buffers for TRT inference on CPU and GPU
        host_mem = cuda.pagelocked_empty(size, dtype) #NumPy array in fixed RAM.
        #Page-locked memory is fixed physical RAM location, Normal CPU memory can be moved around by OS/python.
        #Pinned memory allows much faster transfers between CPU and GPU as it doesn't require a temporary buffer to locate the data.
        device_mem = cuda.mem_alloc(host_mem.nbytes) #function inside bracket calculates how many bytes used by host buffer, so same can be allocated on GPU VRAM

        host_buffers[name] = host_mem
        device_buffers[name] = device_mem

        context.set_tensor_address(name, int(device_mem))

    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("Image not found")

    orig_h, orig_w = img.shape[:2]

    input_img = preprocess(img, input_h, input_w)

    np.copyto(host_buffers[input_name], input_img.ravel())  #copy the preprocessed image data into the input buffer [line 81] that TRT will use for inference.
#ravel() flattens a multidimensional array into a 1D array of the total number (multiplied) of elements. So basically, flatten image tensor in one dimension and add it to inputs buffer.

    #INFERENCE
    cuda.memcpy_htod_async(device_buffers[input_name], host_buffers[input_name], stream)#copy image tensor from host buffer [0],[0] to device buffer [0],[1], queued in the stream
    #Execution of neural network in TensorRT GPU, also queued in stream.
    context.execute_async_v3(stream_handle=stream.handle) #Actual inference in GPU. TensorRT 10 removed execute_async_v2
    cuda.memcpy_dtoh_async(host_buffers[output_name], device_buffers[output_name], stream) #only copy the image tensors that are in outputs, queued in the stream
    stream.synchronize() #Wait until all tasks in the stream are finished
    
    # Parse output : host_buffers[1] >>>first output tensor in host buffer
    # Expected output: [N, 6] -> x1, y1, x2, y2, score, class
    output_shape = context.get_tensor_shape(output_name)
    output = host_buffers[output_name].reshape(output_shape)
    output = output.reshape(-1, output.shape[-1]) #reshape() changes the dimensions of a NumPy array without changing the data
    #output tensors shape is of the type (1, 8400, 85)=batch, detections, values per detection. 85 values represent [x, y, w, h, objectness, class 0 score, class 1 score, ....]. 
    #This is standard format because YOLO is trained on COCO which has 80 classes. [-1] tells numpy to automatically calculate this dimension, because custom models have variety of numbers of classes
    #So NumPy determines how many rows are needed to keep the total number of elements unchanged after reshaping. After reshaping the result is (8400, 85).
    #This is done because for post-processing it is easier to work with tensors without the batch dimension.
    #Basically this line flattens all dimensions except the last one so the tensor becomes a 2D array where each row represents one detection and each column represents prediction attributes.
    
    print("Output shape:", output.shape)
    print("Max value:", np.max(output))

    boxes = [] #lists, they are simple, lightweight, ordered, mutable, duplicate values, indexed, dynamic sizes, nested but slow. 
    scores = []  #So Images are stored as np arrays
    class_ids = [] ## arrays are used instead when we need vectorized math. For ex: IoU comparison, NMS acceleration, box transformations

    for det in output:
        x, y, w, h = det[:4]
        obj_conf = det[4]
        class_scores = det[5:] #this will contain 2 values because my model has 2 classes

        cls_id = np.argmax(class_scores) #Returns index of the max value. Which class has highest probabilty of being that object
        cls_conf = class_scores[cls_id] 

        score = obj_conf * cls_conf
        #Set confidence thresholds
        if score < 0.3:
            continue

        # Scale boxes back to original image
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        x1 = int((x - w / 2) * scale_x)
        y1 = int((y - h / 2) * scale_y)
        x2 = int((x + w / 2) * scale_x)
        y2 = int((y + h / 2) * scale_y)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(score))
        class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.45) #inbuilt cv2 function for NMS

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{class_ids[i]} {scores[i]:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display
    cv2.imshow("YOLOX TensorRT - Jetson", img)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  #Clean practice to use this function calling in python.
#When a python file runs, python sets __name__ = "__main__".
#When the file is imported as a module(import script) then __name__ = "script".
#so structuring the code like this in main() and calling ensures that the code runs only when the file is executed directly.

