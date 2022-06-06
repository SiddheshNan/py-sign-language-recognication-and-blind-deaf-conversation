import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras
import numpy as np
import cv2
import pyttsx3
import threading

engine = pyttsx3.init()
frames_to_skip = 20
actions = ['all the best',
           'i love you',
           'bathroom',
           'thank you',
           'promise',
           'how are you',
           'goodbye',
           'okay',
           'stop',
           'hello',
           'n/a']

print("loading...")
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
prev_text = ''


def speak_thread(text):
    global engine, prev_text
    try:
        if text == 'n/a' or text == prev_text:
            return
        prev_text = text
        engine.say(text)
        engine.runAndWait()
    except:
        pass


def speak(text):
    x = threading.Thread(target=speak_thread, args=(text,), daemon=True)
    x.start()


def detect(img):
    img = cv2.resize(img, (224, 224))
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    # print(prediction)
    prediction_new = prediction[0].tolist()
    detected_action = prediction_new.index(max(prediction_new))
    # print("detected action: " + actions[detected_action])
    detected_acc = max(prediction_new)
    # print("accuracy: " + str(detected_acc))
    return actions[detected_action], str(round(detected_acc, 2))


def do_start():
    webcam = cv2.VideoCapture(0)
    ii = 0
    text = ''
    cv2.namedWindow("Image")
    while True:
        _, frame = webcam.read()
        # frame = cv2.flip(frame, 1, 1)
        cv2.putText(
            frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(10)
        ii += 1
        if ii >= frames_to_skip and key:
            (det_action, det_accu) = detect(frame)
            text = det_action + " - " + det_accu + "%"
            speak(det_action)
            ii = 0
        if cv2.waitKey(20) % 256 == 27:
            print("Esc Pressed.. Exiting..")
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    do_start()
