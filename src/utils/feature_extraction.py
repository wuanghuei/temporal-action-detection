import cv2
import numpy as np


def extract_pose_features(frame, pose_detector):
    results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #BGR → RGB

    if not results.pose_landmarks: #1 container chứa các landmark
        return np.zeros(33 * 3) # no landmark, trả về mảng toàn 0 (33 × 3)

    pose_landmarks = [] #33 landmarks * 3 Direction
    for landmark in results.pose_landmarks.landmark: #instance / list 
        pose_landmarks.extend([landmark.x, landmark.y, landmark.z])

    return np.array(pose_landmarks)


def compute_velocity(pose_data):
    T, D = pose_data.shape
    velocity = np.zeros_like(pose_data)

    if T > 1:
        velocity[1:] = pose_data[1:] - pose_data[:-1]

    max_abs_val = np.max(np.abs(velocity)) + 1e-10
    velocity = velocity / max_abs_val

    return velocity
