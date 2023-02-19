# 2/5/23
# Going to try and run live webcam object detection 
# 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import time

from PIL import Image
from six import BytesIO


import tensorflow_hub as hub


#sys.path.append('/Users/charleshood/Documents/Github/daxon/python-macos/models/research')

#from object_detection.utils import visualization_utils as viz_utils
print(tf.__version__)
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

label_path = './coco17_labels.txt'
model = './ssd_mobilenet_v1_1_metadata_1.tflite' #'./object_detection_mobile_object_localizer_v1_1_default_1.tflite' 
enable_edgetpu = False
num_threads = 1

# Import the label data
label = []
with open(label_path, "r") as f:
    for line in f.readlines():
        label.append(line.rstrip())

# Start up the interpreter
interpreter = tf.lite.Interpreter(model_path=model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('input:',input_details)
print('output:',output_details)

# Run our own data through a webcam
cap = cv2.VideoCapture(1)

last_time = time.time_ns()
while True:
    # Read frame from camera
    current_time = time.time_ns()
    frame_time = (current_time - last_time)/1000000000
    frame_rate = 1/frame_time
    last_time = current_time

    ret, image = cap.read()
    image_resize = cv2.resize(image,(300, 300))

    #image_data = tf.io.gfile.GFile('./person-walking-on-the-beach-with-a-dog.jpg', 'rb').read()
    #image = Image.open(BytesIO(image_data))
    #image_np = np.array(image.getdata()).reshape((1, 300, 300, 3)).astype(np.uint8)
    #input_image = tf.cast(image_np, dtype=tf.uint8)

    #image = Image.open(r"./person-walking-on-the-beach-with-a-dog.jpg")
    #image = image.resize((300,300))
    image_np = np.array(image_resize).reshape(1,300,300,3).astype(np.uint8)

    #print('Image_np:',image_np.shape)

    interpreter.set_tensor(input_details[0]['index'], image_np)
    # Invoke inference.
    interpreter.invoke()


    # Get the model prediction.
    output_dict = {
        'num_detections': int(interpreter.get_tensor(output_details[3]["index"])),
        'detection_classes': interpreter.get_tensor(output_details[1]["index"]).astype(np.uint8),
        'detection_boxes' : interpreter.get_tensor(output_details[0]["index"]),
        'detection_scores' : interpreter.get_tensor(output_details[2]["index"])
    }

    output_dict['detection_classes'] = output_dict['detection_classes'][0]
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    #print('detection_classes', output_dict['detection_classes'])
    #print('detection_boxes', output_dict['detection_boxes'])
    #print('detection_scores', output_dict['detection_scores'])
    
    # Draw and process
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.65:
            ymin = int(max(1,output_dict['detection_boxes'][i][0] * 300))
            xmin = int(max(1,(output_dict['detection_boxes'][i][1] * 300)))
            ymax = int(min(300,(output_dict['detection_boxes'][i][2] * 300)))
            xmax = int(min(300,(output_dict['detection_boxes'][i][3] * 300)))

            cv2.rectangle(image_resize, (xmin,ymin), (xmax,ymax), (10,255,0), 1)
            object_name = label[int(output_dict['detection_classes'][i])]
            image_label = '%s: %d%%' % (object_name, int(output_dict['detection_scores'][i]*100))
            cv2.putText(image_resize, image_label, (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10,255,0), 1) 
            cv2.putText(image_resize, 'fps:'+str(int(frame_rate)), (1,7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)


    # Display output
    cv2.imshow('object detection', image_resize)

    if cv2.waitKey(25) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()