import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Indici dei punti di riferimento delle punte delle dita
finger_tips_indices = [4, 8, 12, 16, 20]

# Apri la webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flippa l'immagine orizzontalmente
        image = cv2.flip(image, 1)

        # Converti l'immagine in RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Rilevamento volti
        face_results = face_detection.process(image)
        
        # Rilevamento mani
        hand_results = hands.process(image)
        
        # Torna a BGR per la visualizzazione
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Disegna le annotazioni sui volti
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(image, detection)
        
        # Disegna le annotazioni sulle mani e memorizza le coordinate in una matrice
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Disegna i pallini sulle punte delle dita
                finger_tips_coords = []
                for index in finger_tips_indices:
                    landmark = hand_landmarks.landmark[index]
                    h, w, _ = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    finger_tips_coords.append((cx, cy))
                    cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                
                # Calcola la variabile in base alla distanza tra i pallini
                distance_sum = 0
                for i in range(len(finger_tips_coords)):
                    for j in range(i + 1, len(finger_tips_coords)):
                        distance_sum += np.linalg.norm(np.array(finger_tips_coords[i]) - np.array(finger_tips_coords[j]))
                
                # Normalizza la variabile
                max_distance = np.linalg.norm(np.array([w, h]))
                variable = distance_sum / (10 * max_distance)  # Normalizza tra 0 e 1

                # Visualizza la variabile sulla fotocamera
                cv2.putText(image, f'Variable: {variable:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Face and Hand Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()