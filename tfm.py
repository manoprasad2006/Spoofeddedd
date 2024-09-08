import cv2
import numpy as np
import tensorflow as tf

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='tflite_quant_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

rp, fp = 0, 0

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
            resized_face = resized_face.astype("float32") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)

            # Set the tensor to the input data
            interpreter.set_tensor(input_details[0]['index'], resized_face)

            # Run the interpreter
            interpreter.invoke()

            # Get the output prediction
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            print(preds)
            

            # Label and draw bounding box
            if preds > 0.5:
                label = 'spoof'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                fp += 1
            else:
                label = 'real'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                rp += 1
        
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

# Display access granted/try again
height, width = 500, 800  # dimensions 
blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background
text = "Access Granted" if rp > fp else "Try Again"
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

print("real", rp, "fake", fp)


