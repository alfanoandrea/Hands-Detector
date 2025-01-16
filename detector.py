import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
from time import sleep

PIN_MOTORE_X = 17
PIN_MOTORE_Y = 18

# Configurazione dei pin per i servomotori
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_MOTORE_X, GPIO.OUT)  # Pin per servo X
GPIO.setup(PIN_MOTORE_Y, GPIO.OUT)  # Pin per servo Y

servo_x = GPIO.PWM(PIN_MOTORE_X, 50)  # Frequenza 50Hz
servo_y = GPIO.PWM(PIN_MOTORE_Y, 50)

servo_x.start(7.5)  # Posizione centrale
servo_y.start(7.5)

# Funzione per muovere i servomotori in base alle coordinate

def move_servos(relative_x, relative_y, center_x, center_y):
    # Converti le coordinate in valori di duty cycle per il servo
    duty_x = 7.5 + (relative_x / center_x) * 2.5  # Regola il range di movimento X
    duty_y = 7.5 + (relative_y / center_y) * 2.5  # Regola il range di movimento Y

    servo_x.ChangeDutyCycle(max(2.5, min(12.5, duty_x)))
    servo_y.ChangeDutyCycle(max(2.5, min(12.5, duty_y)))
    sleep(0.05)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape
        center_x, center_y = w // 2, h // 2

        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(image_bgr, detection)
                bbox = detection.location_data.relative_bounding_box
                face_x = int(bbox.xmin * w + bbox.width * w / 2)
                face_y = int(bbox.ymin * h + bbox.height * h / 2)

                relative_x = face_x - center_x
                relative_y = face_y - center_y

                # Muovi i servomotori per seguire il volto
                move_servos(relative_x, relative_y, center_x, center_y)

                cv2.putText(image_bgr, f'Face Position: ({relative_x}, {relative_y})', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Tracking', image_bgr)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
servo_x.stop()
servo_y.stop()
GPIO.cleanup()
