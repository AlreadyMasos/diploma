import cv2
import numpy as np 
import mediapipe as mp
import time
import counter_class
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

 
helper = counter_class.AngleCounter
counter = 0
pushup_counter = 0
squats_counter = 0
curls_counter = 0
stage = None
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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            angle_shoulder = helper.calculate_angle(shoulder,elbow,wrist)
            angle_knee_l = helper.calculate_angle(hip_l,knee_l,ankle_l)
            angle_knee_r = helper.calculate_angle(hip_r,knee_r,ankle_r)
            cv2.putText(image, str(round(angle_shoulder)), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 10), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(round(angle_knee_l)), 
                           tuple(np.multiply(knee_l, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 10), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(round(angle_knee_r)), 
                           tuple(np.multiply(knee_r, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 10), 2, cv2.LINE_AA
                                )

            if angle_shoulder > 160:
                stage = 'down'
            if angle_shoulder < 30 and stage == 'down':
                stage = 'up'
                curls_counter += 1
            
            if angle_shoulder > 160:
                stage = 'down'
                startp = shoulder[0]
            if angle_shoulder < 100 and stage == 'down' and shoulder[0] <= startp - 0.01:
                stage = 'up'
                pushup_counter += 1
            
            if angle_knee_l > 160 and angle_knee_r > 160:
                stage = 'down'
            if angle_knee_r <= 105 and angle_knee_l <= 106 and stage == 'down':
                stage = 'up'
                squats_counter += 1
        except:
            pass
        #cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # отображение кол-ва повторений
        cv2.putText(image, 'reps', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(curls_counter) + ' curls', 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, str(squats_counter) + ' squats', 
                    (10,100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, str(pushup_counter) + ' pushups', 
                    (10,140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # отображение текущий стадии скручивания
        mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(220,117,65),thickness=2,circle_radius=2),
                                mp_drawing.DrawingSpec(color=(220,60,120),thickness=2,circle_radius=2))

         
        cv2.imshow('DIPLOMA', image)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print(counter)