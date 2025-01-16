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

        # Ottieni le dimensioni dell'immagine
        h, w, _ = image.shape
        center_x, center_y = w // 2, h // 2

        # Disegna le annotazioni sui volti e traccia la posizione del volto
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(image, detection)
                
                # Ottieni le coordinate del volto
                bbox = detection.location_data.relative_bounding_box
                face_x = int(bbox.xmin * w + bbox.width * w / 2)
                face_y = int(bbox.ymin * h + bbox.height * h / 2)
                
                # Calcola la posizione del volto rispetto al centro della fotocamera
                relative_x = face_x - center_x
                relative_y = face_y - center_y
                
                # Visualizza le coordinate relative del volto
                cv2.putText(image, f'Face Position: ({relative_x}, {relative_y})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Disegna le annotazioni sulle mani e memorizza le coordinate in una matrice
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Disegna i pallini sulle punte delle dita
                finger_tips_coords = []
                for index in finger_tips_indices:
                    landmark = hand_landmarks.landmark[index]
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

        # Conta il numero di persone rilevate
        num_people = 0
        if face_results.detections:
            num_people = len(face_results.detections)
        
        # Visualizza il numero di persone in alto a destra
        cv2.putText(image, f'People: {num_people}', (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()