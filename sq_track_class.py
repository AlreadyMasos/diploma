import cv2
import numpy as np 
import mediapipe as mp
import counter_class
import time
def all_track(filename):
    helper = counter_class.AngleCounter
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    counter_sq = 0
    counter_push = 0
    stage_push = None
    stage_knee = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
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
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]


                    angle_knee_left = helper.calculate_angle(hip_left,
                                                        knee_left,
                                                        ankle_left)
                    
                    angle_knee_right = helper.calculate_angle(hip_right,
                                                        knee_right,
                                                        ankle_right)
                    
                    if angle_knee_left > 160 and angle_knee_right > 160:
                        stage_knee = 'down'
                        starter = hip_left[1]
                    if (angle_knee_left <= 130 and angle_knee_right <= 130
                        and stage_knee == 'down' and hip_left[1] >= starter + 0.1):
                        stage_knee = 'up'
                        counter_sq += 1

                    angle_wrist = helper.calculate_angle(shoulder,elbow,wrist)

                    if angle_wrist > 160:
                        stage_push = 'down'
                        startp = shoulder[0]
                        
                    if angle_wrist < 100 and stage_push == 'down' and shoulder[0] <= startp - 0.05:
                        
                        stage_push = 'up'
                        counter_push += 1

                        
                except:
                    pass
                
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    return (counter_push, counter_sq)


