import numpy as np
import cv2
import pyttsx3
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#cap = cv2.VideoCapture("https://192.168.178.133:8080/video")
cap= cv2.VideoCapture(0)

x=20
y=400
Faces=[]
Eyes=[]
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    Faces.append(faces)
    if len(Faces)>3:
        if isinstance(faces, tuple) and isinstance(Faces[-3],tuple):
            text="regarder devant."
            cv2.putText(img,text,(x-100,y-100),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,255),3)
            #cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            
            # Initialiser le moteur pyttsx3
            engine = pyttsx3.init()
            
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
            engine.say("veuillez regarder vers l'avant s'il vous plait.")

            # Lancer la conversion et lire le texte
            engine.runAndWait()
        

        for (x,y,w,h) in faces:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # detection des yeux
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 4)

            Eyes.append(eyes)
            #print(type(eyes))
            if len (Eyes)>3:
                if isinstance(eyes, tuple) and isinstance(Eyes[-3], tuple):
                    text="Alert!!!!!!!!!!"
                    cv2.putText(img,text,(x-100,y-60),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0, 255),3)
                    
                    #cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
                    
                    # Initialiser le moteur pyttsx3
                    engine = pyttsx3.init()
                    
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
                    engine.say('veuillez ouvrir les yeux')

                    # Lancer la conversion et lire le texte
                    engine.runAndWait()

                    timestamp = time.time()

                    # Convertir le timestamp en struct_time (local time)
                    local_time = time.localtime(timestamp)

                    # Afficher l'heure au format "HH:MM:SS"
                    formatted_time = time.strftime("%H:%M:%S", local_time)
                    print(f'les yeux fermés à {formatted_time}')

            #for (ex,ey,ew,eh) in eyes:
                
                #cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
                
            
    cv2.imshow('driving_help',img)
    if cv2.waitKey(10)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

