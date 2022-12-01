#import libraries

import numpy as np
import cv2
import tensorflow as tf
from scipy.special import softmax #we need this library for the step 3 of the deep learning

#load all models first is the face detection model and we use the caffe model

face_detection_model = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt','models/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# the second model is the face mask recognition model
model = tf.keras.models.load_model('face_cnn_model')

# label 
labels = ['Mask', 'No Mask', 'Covered Mouth Chin', 'Covered Nose Mouth']

#now we should choose a color for each label
#mask=green 0,255,0
#no mask=red 255,0,0
#covered mouth chin=blue 0,0,255
#covered nose mouth=pink 255,0,255

def getColor(label):
    if label == "Mask":
        color = (0,255,0)

    elif label == 'No Mask':
        color = (0,0,255)
    elif label == 'Covered Mouth Chin':
        
        color = (255,0,0)
    else:
        color = (255,0,255)

    return color

#getColor('Mask') #show the color we choose for mask
#getColor('No Mask') #show the color we choose for no mask
#getColor('Covered Mouth Chin') #show the color we choose for covered mouth chin
#getColor('Covered Nose Mouth') #show the color we choose for covered nose mouth

#recognition part code

def face_mask_prediction(img):
    #step 1: face detection
    image = img.copy()
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,117,123),swapRB=True)

    face_detection_model.setInput(blob)
    detection = face_detection_model.forward() # it will give us the detection
    for i in range(0,detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.5:
            box = detection[0,0,i,3:7]*np.array([w,h,w,h])
            box = box.astype(int)
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            #cv2.rectangle(image,pt1,pt2,(0,255,0),1) #the rectangle box around the face

            #step 2: data prepeocessing
            #need to crop the face

            face = image[box[1]:box[3],box[0]:box[2]]
            face_blob = cv2.dnn.blobFromImage(face,1,(224,224),(104,117,123),swapRB=True)
            face_blob_squeeze = np.squeeze(face_blob).T #for correct rotation .T
            face_blob_rotate = cv2.rotate(face_blob_squeeze,cv2.ROTATE_90_CLOCKWISE) #for correct structure 
            face_blob_flip = cv2.flip(face_blob_rotate,1) #for flip the image

            # normalization

            img_norm = np.maximum(face_blob_flip,0)/face_blob_flip.max()



            #step 3: deep learning (cnn)

            img_input = img_norm.reshape(1,224,224,3)
            result = model.predict(img_input)
            #print(result) #the probabilities of the labels
            result = softmax(result)[0]
            confidence_index = result.argmax() #take out the labels out of this and exctract only where we have the highest value and that means that it wears a mask
            print('The confidence score is =',confidence_index)
            confidence_score = result[confidence_index]
            label = labels[confidence_index] #label out
            label_text = '{}: {:,.0f} %'.format(label,confidence_score*100) #these will print only the integer values
            #print(label_text) #shows if it wears a mask or no and the probability of this


            # put the ractangular box and whow the label on top of the face

            color = getColor(label)
            cv2.rectangle(image,pt1,pt2,color,1)
            cv2.putText(image,label_text,pt1,cv2.FONT_HERSHEY_PLAIN,2,color,2) #thickness is 2 to be more clear the text on top of the face

    return image


