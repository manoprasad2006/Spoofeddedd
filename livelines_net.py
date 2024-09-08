import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load anti-spoofing model weights
model.load_weights('finalyearproject_antispoofing_model_98-0.942647.h5')
print("Model loaded from disk")
rp,fp=0,0
# Initialize webcam
video = cv2.VideoCapture(1)
while True:
    try:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = frame[y-5:y+h+5, x-5:x+w+5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)

            # Predict with the anti-spoofing model
            preds = model.predict(resized_face)[0]
            print(preds)

            # Label and draw bounding box
            if preds > 0.5:
                label = 'spoof'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                fp=fp+1
            else:
                label = 'real'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                rp=rp+1
        
        # Show video
        cv2.imshow('frame', frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    except Exception as e:
        print("Error: ", str(e))
        pass

video.release()
cv2.destroyAllWindows()
height, width = 500, 800  # dimensions 
blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background
if(rp>fp):
 text = "Access Granted"
else:text="Try Again" 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 0, 0)  # Black color text
thickness = 2
position = (50, 250)  # Starting point for the text

# Add text to the image
cv2.putText(blank_image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

# Show the image
cv2.imshow("Blank Page with Text", blank_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
print("real",rp,"fake",fp)
