from keras.models import load_model
import keras
import cv2
import pickle
import numpy as np
classifier = load_model("res/models/parking_bw_32_2.h5")
#video = cv2.VideoCapture(r"res\\videos\\video.mp4")
#video = cv2.VideoCapture(r"t1_trim.mp4")
frame = cv2.imread('res/images/screenshot.png')
#print(img)
#print(video)
with open("res/pickles/roi","rb") as fp:
    rois = pickle.load(fp)
rois = rois[1:-1]

print(rois)
#while True:
    # my addition
empty = 0
full = 0
#ret, frame = img.read()
if frame.any():
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(frame_bw)
    frame_bw = frame_bw / 255
    for roi in rois:
        parking_spot = frame_bw[roi[0]:roi[1], roi[2]:roi[3]]
        try:
            spot_resized = cv2.resize(parking_spot, (32, 32))
            spot_reshaped = spot_resized.reshape(1,32,32,1)
        
            predicted = classifier.predict_classes(spot_reshaped)
            status = int(predicted[0][0])
            if(status==0): #spot full
               cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0, 0, 255), 2)
               full+=1
            else: #spot empty
                cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 2)
                empty+=1
            cv2.imshow('frame', frame)
        except Exception as e:
            continue
           # print(str(e))
            
print('right : ' + str(full))
print('wrong : ' + str(empty))
#if cv2.waitKey(20) & 0xFF == ord('q'):
 #   break

#video.release()
#cv2.destroyAllWindows()
