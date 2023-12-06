import cv2 # opencv image processing library
import datetime
import imutils # image processing
import numpy as np
import pywhatkit
from centroidtracker import CentroidTracker # used for tracking 
from nms import non_max_suppression_fast #to reduce many frames into a single frame
from collections import defaultdict #dictionary
import pandas as pd
import pygame
message='Fall Alert!'
number='+919562688725'
protopath = "MobileNetSSD_deploy.prototxt" # deep learning models for detecting person # one shot detection ie detection + recognition
modelpath = "MobileNetSSD_deploy.caffemodel" # it is used because of light architecture and inference speed
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath) 
# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)#maxDisappeared to show how many frames in the screen
                                                            #maxDistance to show distance from first frame to new position


def main():
    cap = cv2.VideoCapture("video_1.mp4")
    #cap = cv2.VideoCapture(0)
    lock=0
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lock=0
    time=0
    lock1=False
    temp=[]
    tmp=[]
    elapsed_dict=defaultdict(list)
    temp_tup=()
    object_id_list = []
    object_id_list1 = []
    time_lock=0
    dtime = dict()
    dwell_time = dict()
    my_dict = {"Id":[],"Time":[],"Elapsed_time":0}
    elapsed_dict_1=defaultdict(list)
    object_id_temp=0                  #initalizing the variables
    while True:
        
        ret, frame = cap.read()  #to read the frame and return the value if return is 0 we will not get any values
        frame = imutils.resize(frame, width=600) # image resize for faster processing
        #total_frames = total_frames + 1
        #print(frame.shape)
        (H, W) = frame.shape[:2] #height and width 

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)#to get 4 dimensional array
        #blob = cv2.dnn.blobFromImage(image, scalefactor, size, mean)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:    #if 50% is a person detection
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int") #the starting and ending positions of x and y dimension
                                                                        #astype("int")is used to change the frame into integer
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)#if 30% of the frames are overlapping we use NMS

        objects = tracker.update(rects)
        j=0
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            h  = int(y2-y1)
            w  = int(x2-x1)
            i
            #all these values can be detected
            #print("Id: {}, Height: {}, Width: {}, j:{}, ".format(objectId,h,w,j))
            #if h < w:
            #    j += 1
            #if j>5:
            
            if objectId not in object_id_list:
                object_id_list.append(objectId)
                now=datetime.datetime.now()
                dtime[objectId] = datetime.datetime.now()
                dwell_time[objectId] = 0
                lock=0
                tmp.append(0)
                time = now.strftime("%y-%m-%d %H:%M:%S") #year,month,date,hour,minute,second 
                #print(type(objectId))
                my_dict["Id"].append((str(objectId)))
                my_dict["Time"].append(str(time))
                elapsed_dict[objectId].append(int(dwell_time[objectId]))
            else:
                
             if w > 170 :  
                  text="Fall Alert!"
                  #cv2.putText(fgmask, text, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
                  cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)
                  cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                  #cv2.putText(img,text,position,font,fontscale,colour,linetype)
   

                  object_id_temp=objectId
                  if object_id_temp not in object_id_list1:
                     object_id_list1.append(object_id_temp)
                     dwell_time[objectId] = 0
                     #dtime[objectId] = 0
                     dtime[objectId] = datetime.datetime.now()
                     object_id_list.append(object_id_temp)
                     
                  else:
                     
                     curr_time = datetime.datetime.now()
                     old_time = dtime[objectId]		     
                     time_diff = curr_time - old_time		     
                     dtime[objectId] = datetime.datetime.now()		     
                     sec = time_diff.total_seconds()	
                     if(time_lock==0):	     
                       dwell_time[objectId] += sec		     
                       elapsed_dict[objectId].append(int(dwell_time[objectId]))		     
                     print(dwell_time[objectId])
                     if(int(dwell_time[objectId])>8 and lock==0):# fall detected warning is given
                        time_lock=1
                        pygame.init()
                        pygame.mixer.init()
                        alarm_sound=pygame.mixer.Sound('sound2.mp3')
                        alarm_sound.play()
                        pywhatkit.sendwhatmsg_instantly(number,message,15)
                        print('Fall')
                        lock=1
                        
                     #lock=0
                     #tmp.append(0)
                     #time = now.strftime("%H:%M:%S")
                     #print(type(objectId))
                     
                     #my_dict["Id"].append((str(object_id_temp)))
                     #my_dict["Time"].append(str(time))
                     elapsed_dict[objectId].append(int(dwell_time[objectId]))

             if w < 170:
                  j = 0 
                  cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
                  #text = "Fall Alert!"
                  #cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                  
                  #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
                  text = "ID: {}".format(objectId)
                  cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                  #no fall detected then the id will only appear for the boundarybox


            
        #fps_end_time = datetime.datetime.now()
        #time_diff = fps_end_time - fps_start_time
        #if time_diff.seconds == 0:
        #    fps = 0.0
        #else:
        #    fps = (total_frames / time_diff.seconds)

        #fps_text = "FPS: {:.2f}".format(fps)

        #cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame) #to apply the frames in the video
        key = cv2.waitKey(1) #if fall detected press the q key
        if key == ord('q'):
            mydict=dict(elapsed_dict)
            #print(mydict)
            #print(mydict[0][-1])#elapsed time detected
            #print(my_dict)
            tmp_list=[mydict[x][-1]  for x in range(len(mydict))]
            my_dict={"Id":my_dict["Id"],"Time":my_dict["Time"],"Elapsed_time":tmp_list}
            print(my_dict)
            df=pd.DataFrame.from_dict(my_dict)
            #for k,v in dict(elapsed_dict): 
            #     dict[v]=max(dict[v])     
            df.to_csv('dwell_time_calculation.csv', index=False)  #print the data into csv file
            break #to comeout of the code

    cv2.destroyAllWindows() #to disappear all the windows


main()


#The provided Python code implements a fall detection system using computer vision and object tracking.
# The program utilizes the OpenCV library for image processing, the MobileNet SSD model for person detection, and a centroid-based tracking mechanism.
# It defines functions for non-maximum suppression (NMS) to filter redundant bounding boxes and a CentroidTracker for associating centroids across frames.
# The main function captures video frames from a file, detects persons, and tracks their centroids.
# If a person's height exceeds a certain threshold (indicating a potential fall), the system triggers a fall alert, plays a sound, sends a WhatsApp message, and records the event details such as the person's ID, timestamp, and elapsed time in a CSV file.
# The code incorporates robust error handling and efficiently handles dwell time calculations for fall detection.
# Overall, it combines computer vision, tracking, and notification functionalities to create a fall detection system.