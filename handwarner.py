import cv2
import numpy as np 
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pTime = 0
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            landmarks = results.pose_landmarks.landmark
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            hands = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX_KNUCKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_INDEX_KNUCKLE.value].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_INDEX_KNUCKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_INDEX_KNUCKLE.value].y]
            
            face = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y,
                    landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,
                    landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y,
                    landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,
                    landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y
                    ]
            print(face)
            print(hands)
            for point in hands:
                for point1 in face:
                    print(point1,point)
                    if round(point,1) == round(point1,1):
                        cv2.putText(image,'NO HANDS NEAR FACE',
                        tuple(np.multiply(nose, [640,480]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5,(100,120,80),2,cv2.LINE_AA)
                    
        except:
            pass
        #cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(220,117,65),thickness=2,circle_radius=2),
                                mp_drawing.DrawingSpec(color=(220,60,120),thickness=2,circle_radius=2))
        cv2.imshow('warner', image)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(fps), (100,100),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
