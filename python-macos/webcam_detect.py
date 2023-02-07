# 2/5/23
# Going to try and run live webcam object detection 
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

import tensorflow_hub as hub


#sys.path.append('/Users/charleshood/Documents/Github/daxon/python-macos/models/research')

#from object_detection.utils import visualization_utils as viz_utils
print(tf.__version__)
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))


model = './lite-model_ssd_mobilenet_v2_fpn_100_fp32_default_1.tflite'
enable_edgetpu = False
num_threads = 1

interpreter = tf.lite.Interpreter(model_path=model)
interpreter.allocate_tensors()

print(interpreter.get_signature_list())

# Run our own data through a webcam
cap = cv2.VideoCapture(1)

while True:
    # Read frame from camera
    ret, image_np = cap.read()
    image_np_resize = cv2.resize(image_np,(320, 320))

    input_image = tf.cast([image_np_resize], dtype=tf.float32)
    print(input_image)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('input:',input_details)
    print('output:',output_details)
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    result = interpreter.get_tensor(output_details[0]['index']) #Gets box predictor
    print(result.shape)
    print(result)
    
    # Image has dimensions of [x, y, 3], reduce to greyscale, default shrinks dimensions
    #image_np_grey = np.mean(image_np_resize, axis=2) / 255.0

    #img = (np.expand_dims(image_np_grey, axis=0))
    
    #predictions_single = probability_model.predict(img)
    #print(class_names[np.argmax(predictions_single[0])])


    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    #image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    #input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    #detections, predictions_dict, shapes = detect_fn(input_tensor)

    #label_id_offset = 1
    #image_np_with_detections = image_np.copy()


    # Display output
    cv2.imshow('object detection', image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()