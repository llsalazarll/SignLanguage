import cv2
import pickle
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p' , 'rb') )
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False , max_num_hands=1 , min_detection_confidence=0.3)
labels_dict = {0:'A' , 1 : 'B' , 2: 'L'}

while True :
    ret, frame = cap.read()
    data_aux=[]
    x_=[]
    y_=[]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W , _ = frame.shape

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame ,
                hand_landmarks ,
                mp_hands.HAND_CONNECTIONS ,
                mp_drawing_style.get_default_hand_landmarks_style() ,
                mp_drawing_style.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                data_aux.append(x)
                data_aux.append(y)
                data_aux.append(z)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        pred = model.predict([np.asarray(data_aux)])

        pred_char = labels_dict[int(pred[0])]


        cv2.rectangle(frame , (x1,y1) , (x2,y2) , (0,0,0) , 4 )
        cv2.putText(frame, pred_char, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('FFRAME' , frame)
    if cv2.waitKey(25) & 0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()