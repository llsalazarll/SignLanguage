import os
import mediapipe as mp

mp_path = os.path.dirname(mp.__file__)

for item in os.listdir(mp_path):
    print(item)


print(hasattr(mp, "solutions"))