import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, Response, render_template
import time
import threading

class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Error: Unable to access the camera.")
            exit()
        self.model = load_model('SLDmodel.h5')  # Load your trained model
        self.index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']  # List of characters for prediction
        self.y = None
        self.last_prediction_time = time.time()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            print("Failed to grab frame")
            return None
        frame = cv2.resize(frame, (640, 480))  # Resize the frame
        copy = frame.copy()  # Define Region of Interest (ROI) for sign language
        copy = copy[150:150+200, 50:50+200]  # Save the ROI as an image and make predictions

        current_time = time.time()
        if current_time - self.last_prediction_time >= 1.0:  # 1 second delay
            cv2.imwrite('image.jpg', copy)
            copy_img = image.load_img('image.jpg', target_size=(64, 64))  # Resize image to match model input
            x = image.img_to_array(copy_img)
            x = np.expand_dims(x, axis=0)
            pred = np.argmax(self.model.predict(x), axis=1)
            self.y = pred[0]
            self.last_prediction_time = current_time

        cv2.putText(frame, f'The Predicted Alphabet is: {self.index[self.y] if self.y is not None else ""}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("No frame captured.")
            break
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    video = Video()
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
