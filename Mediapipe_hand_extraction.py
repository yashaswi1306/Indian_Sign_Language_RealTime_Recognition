import mediapipe as mp
from PIL import Image
import numpy as np

mp_hands=mp.solutions.hands

def extract_landmarks(img):

    with mp_hands.Hands(
        # creates detector (better to use with like in file handling as it automatically clears up allocated memory after loop ends)
        static_image_mode=True, # fastapi gives static imges as input
        max_num_hands=2,
        min_detection_confidence=0.2
    ) as detector:
        arr=img.convert("RGB")
        arr=np.array(arr)
        res=detector.process(arr)
    
    if not res.multi_hand_landmarks:
        return None
    
    # now each hand has 21 landmarks with x,y,z coords, so 63 in total

    left_coords=np.zeros(63,dtype=np.float32)
    right_coords=np.zeros(63,dtype=np.float32)

    uses_two_hands=0.0

    for l_marks,handedness in zip(res.multi_hand_landmarks,res.multi_handedness):
        #  if two hands then the loop runs twice, else only once

        label=handedness.classification[0].label # Left or Right hand
        coords=[]

        for lm in l_marks.landmark:
            coords.append([lm.x,lm.y,lm.z])

        coords=np.array(coords,dtype=np.float32) # (21,3)
        coords=coords.flatten() # (63,)

        if label=="Left":
            left_coords=coords
        else:
            right_coords=coords

    if res.multi_hand_landmarks and len(res.multi_hand_landmarks)==2:
        uses_two_hands=1.0

    return np.concatenate([[uses_two_hands],left_coords,right_coords]) # [] for uses_ two_hands cuz float to list for concatenation