import cv2
import time
import threading
import numpy as np
import signal # This python module lets you handle OS-level signals (like interrupts, termination requests, etc.)
import sys # The sys module, lets you interact with the python interpreter, the OS. It is a bridge between your code and the system running it.
import PySpin
import tensorrt as trt
import pycuda.driver as cuda
import vpi #Vision Programming Interface
import queue

from flask import Flask, Response
from collections import deque

cuda.init()
device = cuda.Device(0)

# Temporary context to build engine
main_ctx = device.make_context()

# =========================
# CONFIG
# =========================
MODEL_PATH = "yoloxs140.engine"
CONF_THRESHOLD = 0.45
PORT = 5000

INPUT_SIZEH = 672  # YOLOX input resolution
INPUT_SIZEW = 896

# =========================
# GLOBALS
# =========================
t_cam = None
t_inf = None
frame_queue = queue.Queue(maxsize=1)
output_queue = queue.Queue(maxsize=1)

camera_fps = 0
inference_fps = 0
running = True

app = Flask(__name__)

streaming_active = False
active_clients = 0
stream_lock = threading.Lock()

# =========================
# TEMPORAL FILTER
# =========================
TEMP_WINDOW = 5      # frames to remember
TEMP_CONFIRM = 3     # detections needed

person_history = deque(maxlen=TEMP_WINDOW)

# =========================
# LOAD TENSORRT ENGINE
# =========================
print("Loading YOLOX TensorRT engine...")
print("Model loaded.")

# =========================
# INIT FLIR  (UNCHANGED)
# =========================
def init_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        system.ReleaseInstance()
        raise RuntimeError("No FLIR camera detected")

    cam = cam_list[0]
    cam.Init()

    return system, cam_list, cam

bayer_conversion = cv2.COLOR_BAYER_RG2RGB_EA #Edge aware demosaicing

# =========================
# CAMERA THREAD (UNCHANGED)
# =========================
def camera_thread():
    global camera_fps, running

    frame_count = 0
    start_time = time.time()

    while running:
        try:
            image_result = cam.GetNextImage(1000) # An SDK image object (not usable by numpy/openCV)

            if image_result.IsIncomplete():
                image_result.Release()
                continue

            frame = image_result.GetNDArray() # Converts an image from the SDK into a numpy array. It converts from (H,W) to (H,W,3)
            frame = cv2.cvtColor(frame, bayer_conversion)
            image_result.Release()

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # drop old frame
                except queue.Empty:
                    pass
            frame_queue.put(frame)

            frame_count += 1

            if time.time() - start_time >= 1.0:
                camera_fps = frame_count
                frame_count = 0
                start_time = time.time()

        except PySpin.SpinnakerException:
            continue


# =========================
# INFERENCE THREAD (YOLOX TRT)
# =========================

def inference_thread():
    ctx = device.make_context() 
    try:
        global latest_frame, output_frame, inference_fps, running, true_fps

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        with open(MODEL_PATH, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)  #runtime is state of model execution. 
            engine = runtime.deserialize_cuda_engine(f.read())
    
        context = engine.create_execution_context() # Engine is the factory blueprint, context is the running instance of the machine.
        #the context is the object that holds runtime states, binds i/o memory and runs the model. It manages things that change per inference run: input tensor shapes, memory binds, execution states.
        #One engine can have multiple contexts, this allows parallel inference. Good for video pipelines, multi-camera systems, high throughput servers.
        #Basically it does the process: read input GPU memory > run the network > write output GPU memory.
    
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_name = name
            else:
                output_name = name
    
        context.set_input_shape(input_name, (1, 3, INPUT_SIZEH, INPUT_SIZEW))
    
        stream = cuda.Stream()
    
        host_input = cuda.pagelocked_empty( #allocates pinned CPU memory. Pinned memory is faster for CPU -> GPU transfer and avoids extra copies.
            trt.volume((1, 3, INPUT_SIZEH, INPUT_SIZEW)), # 1x3x672x896=1,806,336 elements
            dtype=np.float32
        ) # This creates a CPU-side buffer (host memory) for the input image
        device_input = cuda.mem_alloc(host_input.nbytes) # Allocates GPU memory for input tensor, size is same as host_input and stored in GPU-accessible memory
    
        output_shape = context.get_tensor_shape(output_name)
        host_output = cuda.pagelocked_empty(
            trt.volume(output_shape),
            dtype=np.float32
        )
        device_output = cuda.mem_alloc(host_output.nbytes)
    
        context.set_tensor_address(input_name, int(device_input))
        context.set_tensor_address(output_name, int(device_output))
    
        frame_count = 0
        start_time = time.time()
    
        while running:
    
            try:
                frame = frame_queue.get(timeout=0.01)
            except queue.Empty:
                continue
    
            orig_h, orig_w = frame.shape[:2]
            x1 = y1 = x2 = y2 = np.array([], dtype=int)
            scores = np.array([], dtype=np.float32)
            cls_ids = np.array([], dtype=int)
            final_indices = []
            
            num_classes = 0
            # -------------------------
            # Preprocess
            # -------------------------
            #Trying with VPI resize function, instead of the cv2.resize function.
            with vpi.Backend.CUDA:
                vpi_image = vpi.asimage(frame)
                resized = vpi_image.rescale(
                    size=(INPUT_SIZEW, INPUT_SIZEH), 
                    interp=vpi.Interp.LINEAR, #Use linear interpolation
                    border=vpi.Border.CLAMP #Handle border pixels
                )
                img = resized.cpu()  #bring back to CPU for TRT copy
           
            #Convert an OpenCV image into tensor format for the neural network.
            img = img.astype(np.float32) # Because Neural Nets operate using floating point numbers.
            img = img.transpose(2, 0, 1) # OPenCV stores images in HWC, deep learning frameworks usually expect CHW.
            img = np.expand_dims(img, axis=0) # Here we add a batch dimension, because neural networks process batches of images.
            np.copyto(host_input, img.ravel()) # copy the preprocessed image data into the input buffer [line 152] that TRT will use for inference.
            # ravel() flattens a multidimensional array into a 1D array of the total number (multiplied) of elements. So basically, flatten image tensor in one dimension and add it to inputs buffer.
            
            #Inference
            cuda.memcpy_htod_async(device_input, host_input, stream) # copy image tensor from host buffer [0],[0] to device buffer [0],[1], queued in the stream
            #Execution of neural network in TensorRT GPU, also queued in stream.
            context.execute_async_v3(stream_handle=stream.handle) # Actual inference in GPU. TensorRT 10 removed execute_async_v2
            cuda.memcpy_dtoh_async(host_output, device_output, stream) # only copy the image tensors that are in outputs, queued in the stream
            stream.synchronize() # Wait until all tasks in the stream are finished
    
            output = host_output.reshape(output_shape)
            output = output.reshape(-1, output.shape[-1])  #reshape() changes the dimensions of a NumPy array without changing the data
            #output tensors shape is of the type (1, 8400, 85)=batch, detections, values per detection. 85 values represent [x, y, w, h, objectness, class 0 score, class 1 score, ....]. 
            #This is standard format because YOLO is trained on COCO which has 80 classes. [-1] tells numpy to automatically calculate this dimension, because custom models have variety of numbers of classes
            #So NumPy determines how many rows are needed to keep the total number of elements unchanged after reshaping. After reshaping the result is (8400, 85).
            #This is done because for post-processing it is easier to work with tensors without the batch dimension.
            #Basically this line flattens all dimensions except the last one so the tensor becomes a 2D array where each row represents one detection and each column represents prediction attributes.
    
            # -------------------------
            # Vectorized Postprocess
            # -------------------------
    
            # Sigmoid
            obj = 1.0 / (1.0 + np.exp(-output[:, 4]))
            cls_scores = 1.0 / (1.0 + np.exp(-output[:, 5:]))
    
            cls_ids = np.argmax(cls_scores, axis=1) # Returns index of the max value. Which class has highest probabilty of being that object
            cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
            scores = obj * cls_conf
    
            mask = scores > CONF_THRESHOLD
            output = output[mask]
            scores = scores[mask]
            cls_ids = cls_ids[mask]
    
            num_classes = output_shape[-1] - 5
    
            # COCO model → only person
            if num_classes == 80:
                person_mask = cls_ids == 0
                output = output[person_mask]
                scores = scores[person_mask]
                cls_ids = cls_ids[person_mask]
    
            if len(output) > 0:
    
                boxes = output[:, :4] # lists, they are simple, lightweight, ordered, mutable, duplicate values, indexed, dynamic sizes, nested but slow.
                # Scale boxes back to original image
                scale_x = orig_w / INPUT_SIZEW
                scale_y = orig_h / INPUT_SIZEH
    
                x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
                y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
                x2 = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x
                y2 = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y
    
                x1 = x1.astype(int)
                y1 = y1.astype(int)
                x2 = x2.astype(int)
                y2 = y2.astype(int)
                
                #NMS during Inference code logic here
                #convert to xywh format required by OpenCV NMS
                boxes_xywh = []
                widths = x2 - x1
                heights = y2 - y1
                boxes_xywh = np.stack([x1, y1, widths, heights], axis=1).tolist()
                #Fully vectorized NMS box conversion loop
                final_indices = []
                
                # Class wise NMS
                unique_classes = np.unique(cls_ids)
                
                for cls in unique_classes:
                    cls_mask = cls_ids == cls
                    cls_boxes = np.array(boxes_xywh)[cls_mask].tolist()
                    cls_scores = scores[cls_mask].tolist()
                    
                    indices = cv2.dnn.NMSBoxes(
                        cls_boxes,
                        cls_scores,
                        CONF_THRESHOLD,
                        0.45 #NMS IoU threshold
                    )
                    
                    if len(indices) > 0:
                        for idx in indices.flatten():
                            global_idx = np.where(cls_mask)[0][idx]
                            final_indices.append(global_idx)
    
            if output_queue.full():
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    pass

            output_queue.put((frame, final_indices, x1, y1, x2, y2, scores, cls_ids, num_classes))
                  
            frame_count += 1
            total_frames += 1
            
            if time.time() - start_time >= 1.0:
                inference_fps = frame_count
                frame_count = 0
                start_time = time.time()
                #print(f"Inference FPS: {inference_fps} | Streaming: {streaming_active}")
               
    finally:
        ctx.pop()

#TEMPORAL FILTERING
def temporal_filter(detected):
    """
    detected = True if person detected in frame
    """
    person_history.append(detected)

    if sum(person_history) >= TEMP_CONFIRM:
        return True

    return False
    
# =========================
# STREAM + SHUTDOWN (UNCHANGED)
# =========================

def generate():
    global output_frame, running
    global streaming_active, active_clients

    with stream_lock:
        active_clients += 1
        streaming_active = True

    print("Client connected. Active clients:", active_clients)

    target_fps = 15
    frame_interval = 1.0 / target_fps
    last_sent = 0

    try:
        while running:
            if not streaming_active:
                break

            if output_queue.empty():
                time.sleep(0.001)
                continue

            now = time.time()
            if now - last_sent < frame_interval:
                time.sleep(0.001)
                continue

            try:
                frame, final_indices, x1, y1, x2, y2, scores, cls_ids, num_classes = output_queue.get(timeout=0.01)
                person_detected = len(final_indices) > 0
                confirmed = temporal_filter(person_detected)

                if confirmed:
                    for i in final_indices:
                        #remove later, accomodating for multiple models
                        if num_classes == 2:
                            label_name = "Person" if cls_ids[i] == 0 else "Helmet"
                            color = (255,0,255) if cls_ids[i] == 0 else (255,255,0)
                        else:
                            label_name = "Person"
                            color = (0,255,0)

                        cv2.rectangle(frame,
                                      (x1[i], y1[i]),
                                      (x2[i], y2[i]),
                                      color, 2)

                        cv2.putText(frame,
                                    f"{label_name} {scores[i]:.2f}",
                                    (x1[i], y1[i] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2)
                
                # FPS overlay
                cv2.putText(frame, f"Camera FPS: {camera_fps}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)
                
                cv2.putText(frame, f"Inference FPS: {inference_fps}",
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,255), 2)
                            
                '''cv2.putText(frame, f"True_FPS: {true_fps:.2f}",
                            (10,90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,255), 2)'''
            except queue.Empty:
                continue

            # Encode only when needed
            ret, buffer = cv2.imencode(".jpg", frame)

            if not ret:
                continue

            last_sent = now

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

    finally:
        with stream_lock:
            active_clients -= 1
            if active_clients <= 0:
                streaming_active = False
                active_clients = 0

        print("Client disconnected. Active clients:", active_clients)

@app.route("/video")
def video():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def shutdown_handler(sig, frame):
    safe_shutdown()
    sys.exit(0)
def safe_shutdown():
    global running
    running = False
    time.sleep(0.2)

    if t_cam and t_cam.is_alive():
        t_cam.join(timeout=2)

    if t_inf and t_inf.is_alive():
        t_inf.join(timeout=2)

    try:
        if cam.IsStreaming():
            cam.EndAcquisition()
    except Exception:
        pass

    try:
        cam.DeInit()
    except Exception:
        pass

    try:
        cam_list.Clear()
    except Exception:
        pass

    try:
        system.ReleaseInstance()
    except Exception:
        pass

    try:
        main_ctx.pop()
    except Exception:
        pass

    print("Clean shutdown complete.")
signal.signal(signal.SIGINT, shutdown_handler)


if __name__ == "__main__":  #Clean practice to use this function calling in python.
#When a python file runs, python sets __name__ = "__main__".
#When the file is imported as a module(import script) then __name__ = "script".
#so structuring the code like this in main() and calling ensures that the code runs only when the file is executed directly.
    try:
        system, cam_list, cam = init_camera()
        
        # -------------------------
        # ROI CONFIGURATION
        # -------------------------
        
        ROI_WIDTH = 1408
        ROI_HEIGHT = 1056
        OFFSET_X = 320
        OFFSET_Y = 256

        nodemap = cam.GetNodeMap()

        try:
            cam.EndAcquisition()
        except:
            pass

        node_width = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
        node_height = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
        node_offsetX = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
        node_offsetY = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))

        if PySpin.IsWritable(node_width) and PySpin.IsWritable(node_height):

            node_offsetX.SetValue(0)
            node_offsetY.SetValue(0)

            node_width.SetValue(ROI_WIDTH)
            node_height.SetValue(ROI_HEIGHT)

            node_offsetX.SetValue(OFFSET_X)
            node_offsetY.SetValue(OFFSET_Y)

            print(f"ROI set to {ROI_WIDTH}x{ROI_HEIGHT}")

        else:
            print("WARNING: ROI not writable. Using full sensor.")
        
        # -------------------------
        # Acquisition Settings
        # -------------------------
        node_acq_mode = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionMode")
        )
        node_acq_cont = node_acq_mode.GetEntryByName("Continuous")
        if PySpin.IsWritable(node_acq_mode):
            node_acq_mode.SetIntValue(node_acq_cont.GetValue())
            
        sNodemap = cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(
            sNodemap.GetNode("StreamBufferHandlingMode")
        )
        node_newestonly = node_bufferhandling_mode.GetEntryByName("NewestOnly")
        node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())    

        cam.BeginAcquisition()
        print("FLIR acquisition started.")

        # Start threads
        t_cam = threading.Thread(target=camera_thread, daemon=True)
        t_inf = threading.Thread(target=inference_thread, daemon=True)

        t_cam.start()
        t_inf.start()

        print(f"Stream at http://<JETSON_IP>:{PORT}/video")
        app.run(host="0.0.0.0", port=PORT, threaded=True)

    finally:
        safe_shutdown()
