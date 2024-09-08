import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/finalyearproject_antispoofing_model_mobilenet1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load anti-spoofing model weights
model.load_weights('antispoofing_models/finalyearproject_antispoofing_model_82-1.000000.weights.h5')
print("Model loaded from disk")

# Initialize counters
rp, fp = 0, 0

# Initialize webcam
video = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Continue collecting predictions
        for (x, y, w, h) in faces:
            face = frame[y-5:y+h+5, x-5:x+w+5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            
            preds = model.predict(resized_face)[0]
            print(preds)
            
            # Update counters based on prediction
            if preds > 0.5:
                label = 'spoof'
                fp += 1
            else:
                label = 'real'
                rp += 1
            
            # Draw bounding box and label
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if label == 'real' else (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if label == 'real' else (0, 0, 255), 2)

        # Show video feed
        cv2.imshow('frame', frame)
        
        # Exit on pressing 'q'
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    except Exception as e:
        print("Error: ", str(e))
        continue

# Release video capture and destroy all windows
video.release()
cv2.destroyAllWindows()
