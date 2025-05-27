import cv2
import math
import cvzone
from ultralytics import YOLO

# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")

# Define class names
class_labels = ['Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage', 
'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage', 
'Taillight-Damage', 'bonnet-dent', 'boot-dent', 'doorouter-dent', 'fender-dent', 
'front-bumper-dent', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent']

class_labels_fr =[
    "Bosse sur la carrosserie",
    "Dommage sur le pare-brise avant",
    "Dommage sur le phare avant",
    "Dommage sur la lunette arrière",
    "Bosse sur le marchepied",
    "Dommage sur le rétroviseur latéral",
    "Dommage sur le feu de signalisation latéral",
    "Dommage sur le feu arrière",
    "Bosse sur le capot",
    "Bosse sur le coffre",
    "Bosse sur la porte extérieure",
    "Bosse sur l’aile",
    "Bosse sur le pare-chocs avant",
    "Bosse sur le montant",
    "Bosse sur le panneau arrière latéral",
    "Bosse sur le pare-chocs arrière",
    "Bosse sur le toit"
]


# Load the image
image_path = "Medias/dent10.jfif"
img = cv2.imread(image_path)

# Perform object detection
results = yolo_model(img)

# Loop through the detections and draw bounding boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.3:
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{class_labels_fr[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))
            print(f'{class_labels_fr[cls]} {conf}')
# Display the image with detections
cv2.imshow("Image", img)

# save de image with damage detection
filename='dentdetect10.jpg'
cv2.imwrite(filename, img)

# Close window when 'q' button is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)