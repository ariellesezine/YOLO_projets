import cv2
import numpy as np 
import time
import pyttsx3

net = cv2.dnn.readNetFromDarknet("yolov4.cfg","yolov4.weights")

ln = net.getLayerNames()    
layers_out = [ln[i-1] for i in net.getUnconnectedOutLayers()]
DEFAULT_CONFIDENCE = 0.3
THRESOLD = 0.4


with open ("coco.names", "r") as f:
    LABELS = f.read().splitlines()

cap = cv2.VideoCapture('tracking.mp4')#"http://10.163.207.107:8080/video"
#cap=cv2.VideoCapture("http://10.158.147.77:8080/video")
#cap=cv2.VideoCapture(0)
while True :
    _,image=cap.read()
    start_time=time.time()
    #ret, image = cap.read()
    #print(image.shape)
    #image=cv2.imread()
    height,width,_ = image.shape

    blob= cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0,0,0), 
                                swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layers_out)
    #print(layerOutputs)
    boxes = []
    confidences = []
    classIDs = []
    Areas=[]
    window_name='tracking de personnes sur une video'
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name,800,560)

    for output in layerOutputs:
        
        for detection in output:
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence=scores[classID]
           
            
            if confidence>DEFAULT_CONFIDENCE :
                box=detection[0:4]*np.array([width,height,width,height])
                (center_x,center_y,W,H)=box.astype("int")

                x=int(center_x-(W/2))
                y=int(center_y-(H/2))
                area=W*H
                boxes.append([x,y,int(W),int(H)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
                
            """ if len(Areas)>3 and area>Areas[-3]:
                    output="s'approche de vous"
                    print("-----------------------------------------------------")
                elif len(Areas)>3 and area<Areas[-3]:
                    output="s'eloigne de vous"
                else:
                    output=" " """
                
    indexes=cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIDENCE, THRESOLD)

    colors=np.random.uniform(0,255,size=(len(boxes),3))

    if len(indexes)>0:

        for i in indexes.flatten():
            (x,y,W,H)=boxes[i]
            color=colors[i]
            print(LABELS[classIDs[i]])
            text=" {}".format(LABELS[classIDs[i]])
            cv2.rectangle(image,(x,y),(x+W,y+H),color,2)
            cv2.putText(image, text, (x,y+20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            if classIDs[i]==0:
                
                end_time=time.time()
                img_name=end_time-start_time
                cv2.imwrite(f"Output/{img_name}.jpg",image)
                print(" personne----------------------------------------------")
                
                # Initialiser le moteur pyttsx3
                engine = pyttsx3.init()
                    # Texte à convertir
                text = "une personne détectée?"

                rate = engine.getProperty('rate')
                #print("Volume actuel :", rate)
                engine.setProperty('rate',140)

                # Obtenir le volume actuel
                volume = engine.getProperty('volume')
                #print("Volume actuel :", volume)
                # Définir un nouveau volume
                #engine.setProperty('volume', 1.0) 

                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[0].id)

                # Définir le texte à lire
                engine.say(text)

                # Lancer la conversion et lire le texte
                engine.runAndWait()
                break

    #cv2.imshow("image",image)
    cv2.imshow(window_name,image) 
    if cv2.waitKey(1)==ord("q"):
        break
#cap.release()
cv2.destroyAllWindows()

