import cv2
import numpy as np 
import mediapipe as mp
import counter_class
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

 
helper = counter_class.AngleCounter
counter = 0
stage = None
stage_knee = None

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
            
            knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            angle_knee_left = helper.calculate_angle(hip_left,
                                                    knee_left,
                                                    ankle_left)
                    
            angle_knee_right = helper.calculate_angle(hip_right,
                                                    knee_right,
                                                    ankle_right)
            cv2.putText(image, str(round(angle_knee_left)), 
                           tuple(np.multiply(hip_left, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 10), 2, cv2.LINE_AA
                                )

        
            if angle_knee_left > 160 and angle_knee_right > 160:
                    stage_knee = 'down'
                    starter = hip_left[1]
            if (angle_knee_left <= 130 and angle_knee_right <= 130
                    and stage_knee == 'down' and hip_left[1] >= starter + 0.1):
                    stage_knee = 'up'
                    counter += 1
                
        except:
            pass
        #cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # отображение кол-ва повторений
        cv2.putText(image, 'reps', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # отображение текущий стадии скручивания
        cv2.putText(image, 'stage', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, 
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(220,117,65),thickness=2,circle_radius=2),
                                mp_drawing.DrawingSpec(color=(220,60,120),thickness=2,circle_radius=2))

         
        cv2.imshow('DIPLOMA', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
