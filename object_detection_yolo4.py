import cv2
import numpy as np 

net = cv2.dnn.readNetFromDarknet("yolov4.cfg","yolov4.weights")

ln = net.getLayerNames()    
layers_out = [ln[i-1] for i in net.getUnconnectedOutLayers()]
DEFAULT_CONFIDENCE = 0.5
THRESOLD = 0.4

with open ("coco.names", "r") as f:
    LABELS = f.read().splitlines()

cap = cv2.VideoCapture(0)#"http://10.163.207.107:8080/video"
#cap=cv2.VideoCapture("http://10.229.151.252:8080/video")
while True :
    _,image=cap.read()
    #ret, image = cap.read()

    #image=cv2.imread()
    height,width,_ = image.shape,

    blob= cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0,0,0), 
                                swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layers_out)
    print(layerOutputs)
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        
        for detection in output:
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence=scores[classID]
            
            if confidence>DEFAULT_CONFIDENCE:
                box=detection[0:4]*np.array([width,height,width,height])
                (center_x,center_y,W,H)=box.astype("int")

                x=int(center_x-(W/2))
                y=int(center_y-(H/2))

                boxes.append([x,y,int(W),int(H)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    indexes=cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIDENCE, THRESOLD)

    colors=np.random.uniform(0,255,size=(len(boxes),3))

    if len(indexes)>0:
        for i in indexes.flatten():
            (x,y,W,H)=boxes[i]
            color=colors[i]
            text="{}:{:.4f}".format(LABELS[classIDs[i]],confidences[i])
            cv2.rectangle(image,(x,y),(x+W,y+H),color,2)
            cv2.putText(image, text, (x,y+20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    #cv2.imshow("image",image)
    cv2.imshow("video",image) 
    if cv2.waitKey(1)==ord("q"):
        break
#cap.release()
cv2.destroyAllWindows()

