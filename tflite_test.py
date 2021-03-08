import cv2
import tensorflow as tf
import numpy as np

model_path = r"/home/pi/chain/model.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details= interpreter.get_output_details()
interpreter.allocate_tensors()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


        
def predict_classes(tf_lite_interpreter, y):
    tf_lite_interpreter.set_tensor(input_details[0]['index'], y)
    tf_lite_interpreter.invoke()
    result= tf_lite_interpreter.get_tensor(output_details[0]['index'])
    if result==0:
        print("pass")
    else:
       print("REJECT")


source=cv2.VideoCapture("/home/pi/chain/reject.avi")
while(source.isOpened()):
    ret,img =source.read()
    if ret==True:
        img_resized = cv2.resize(img,(width, height))
        array= np.array(img_resized, dtype=np.float32)
        new= np.expand_dims(array, axis=0)
        input_data=np.vstack([new])
        predict=predict_classes( interpreter, input_data)
        
    cv2.imshow('LIVE',img)
    if cv2.waitKey(10) == ord('q'):
        break
    else:
        break

source.release()  
cv2.destroyAllWindows()
