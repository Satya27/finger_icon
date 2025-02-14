import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


j1_icon = cv2.imread('j1.png', cv2.IMREAD_UNCHANGED)
j2_icon = cv2.imread('j2.png', cv2.IMREAD_UNCHANGED)
j3_icon = cv2.imread('j3.png', cv2.IMREAD_UNCHANGED)

def resize_icon(icon, scale):
    """Resize ikon dengan skala tertentu."""
    new_width = int(icon.shape[1] * scale)
    new_height = int(icon.shape[0] * scale)
    return cv2.resize(icon, (new_width, new_height), interpolation=cv2.INTER_AREA)


j1_icon = resize_icon(j1_icon, 0.7)
j3_icon = resize_icon(j3_icon, 0.7)

def overlay_icon(frame, icon, position):
    """Tampilkan ikon di frame dengan posisi tertentu."""
    h, w, _ = icon.shape
    x, y = position

    if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame

    roi = frame[y:y+h, x:x+w]

    icon_rgb = icon[:, :, :3]
    icon_alpha = icon[:, :, 3] / 255.0

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - icon_alpha) + icon_rgb[:, :, c] * icon_alpha

    frame[y:y+h, x:x+w] = roi
    return frame

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]

                index_y, index_x = int(index_tip.y * frame.shape[0]), int(index_tip.x * frame.shape[1])
                index_dip_y = int(index_dip.y * frame.shape[0])

                middle_y, middle_x = int(middle_tip.y * frame.shape[0]), int(middle_tip.x * frame.shape[1])
                middle_dip_y = int(middle_dip.y * frame.shape[0])

                ring_y, ring_x = int(ring_tip.y * frame.shape[0]), int(ring_tip.x * frame.shape[1])
                ring_dip_y = int(ring_dip.y * frame.shape[0])

        
                index_lifted = (index_dip_y - index_y) > 20
                middle_lifted = (middle_dip_y - middle_y) > 20
                ring_lifted = (ring_dip_y - ring_y) > 20

                if index_lifted:
                    
                    index_icon_pos = (index_x - j1_icon.shape[1] // 2, index_y - j1_icon.shape[0] - 10)
                    frame = overlay_icon(frame, j1_icon, index_icon_pos)

                if middle_lifted:
                
                    middle_icon_pos = (middle_x - j2_icon.shape[1] // 2, middle_y - j2_icon.shape[0] - 10)
                    frame = overlay_icon(frame, j2_icon, middle_icon_pos)

                if ring_lifted:
                    
                    ring_icon_pos = (ring_x - j3_icon.shape[1] // 2, ring_y - j3_icon.shape[0] - 10)
                    frame = overlay_icon(frame, j3_icon, ring_icon_pos)

        cv2.imshow('Finger Icons', frame)

        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()