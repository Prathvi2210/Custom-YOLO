import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Select engine
ENGINE_PATH = "old_data_training/epoch120/yoloxs120.engine"      # custom
#ENGINE_PATH = "yolox_s_standard.engine"         # COCO

IMAGE_PATH = "testImage.jpg"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def preprocess(img, input_h, input_w):
    img_resized = cv2.resize(img, (input_w, input_h))
    img_resized = img_resized.astype(np.float32)
    img_resized = img_resized.transpose(2, 0, 1)
    img_resized = np.expand_dims(img_resized, axis=0)
    return np.ascontiguousarray(img_resized)


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to load engine.")
    return engine


def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # Get IO names
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    input_shape = engine.get_tensor_shape(input_name)
    context.set_input_shape(input_name, input_shape)

    batch, ch, input_h, input_w = input_shape

    stream = cuda.Stream()

    host_buffers = {}
    device_buffers = {}

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        host_buffers[name] = host_mem
        device_buffers[name] = device_mem

        context.set_tensor_address(name, int(device_mem))

    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("Image not found")

    orig_h, orig_w = img.shape[:2]

    input_img = preprocess(img, input_h, input_w)

    np.copyto(host_buffers[input_name], input_img.ravel())
    cuda.memcpy_htod_async(device_buffers[input_name], host_buffers[input_name], stream)

    # Inference
    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(host_buffers[output_name],
                           device_buffers[output_name],
                           stream)
    stream.synchronize()
    
    # Parse output
    output_shape = context.get_tensor_shape(output_name)
    output = host_buffers[output_name].reshape(output_shape)
    output = output.reshape(-1, output.shape[-1])
    
    print("Output shape:", output.shape)
    print("Max value:", np.max(output))

    boxes = []
    scores = []
    class_ids = []

    for det in output:
        x, y, w, h = det[:4]
        obj_conf = det[4]
        class_scores = det[5:]

        cls_id = np.argmax(class_scores)
        cls_conf = class_scores[cls_id]

        score = obj_conf * cls_conf

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

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.45)

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
    main()

