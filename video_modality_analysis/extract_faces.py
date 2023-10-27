# import face_recognition
import os
import cv2
import json
from deepface.DeepFace import analyze
import numpy as np
from tqdm import tqdm
import shutil


# Path to all videos
base_vid_path = "siq2/incomplete_data/video"

# Loading all video names
all_vid_names = os.listdir(base_vid_path)
print("[INFO] Total Videos: ", len(all_vid_names))

# List to store count of faces
all_face_features = {}

# Output path of all images
base_out_path_faces = "faces"
if not os.path.exists(base_out_path_faces):
    os.mkdir(base_out_path_faces)
else:
    shutil.rmtree(base_out_path_faces)
    os.mkdir(base_out_path_faces)

# Looping over every video
for vid_name in tqdm(all_vid_names, desc="Processing Faces for videos"):
    print("[INFO] Starting processing for video: ", vid_name)

    # Complete video path
    abs_vid_path = os.path.join(base_vid_path, vid_name)
    
    # Loading video using cv2
    capture = cv2.VideoCapture(abs_vid_path)
    
    # Check if video opened successfully
    if (capture.isOpened()== False): 
        print("Error opening video stream or file: ", vid_name)
        continue
    
    num_faces = []
    emotions_per_frame = []
    all_face_features[vid_name] = []
    # Looping over all frames in a video to extract faces from all images
    while capture.isOpened():
        
        # Grab a single frame of video
        ret, frame = capture.read()
        
        # Check if frame is correct
        if ret:

            # Analysing faces using Deepface
            face_analysis = analyze(frame)

            # Lists to store all features of faces for all faces in a given frame
            emotions_curr_frame, age_curr_frame, gender_curr_frame, race_curr_frame = [], [], [], []

            # Save all this faces
            for i, face_features in enumerate(face_analysis):

                # Cropping the face
                padding = 10

                # Recognizing features of this face for this particular face
                dominant_emotion = face_features["dominant_emotion"]
                age = face_features["age"]
                gender = face_features["dominant_gender"]
                race = face_features["dominant_race"]
                x, y, w, h = face_features["region"].values()
                face_crop = frame[y: (y + h), x: (x + w)]

                # Appending all features
                emotions_curr_frame.append(dominant_emotion)
                age_curr_frame.append(age)
                gender_curr_frame.append(gender)
                race_curr_frame.append(race)

                # Saving image
                out_abs_face_path = os.path.join(base_out_path_faces, vid_name.split(".")[0] + "_" + str(i) + ".jpg")
                cv2.imwrite(out_abs_face_path, face_crop)

            # Storing face related information
            all_face_features[vid_name].append({
                "num_faces_each_frame": len(face_analysis), 
                "emotions": emotions_curr_frame, 
                "ages": age_curr_frame, 
                "races": race_curr_frame
            })
            

        else:
            print("Error reading frame in video: ", vid_name)

    print("[INFO] Finished processing video: ", vid_name)

# Creating a text file to store all this information
base_txt_file_path = "vid_frame_count.json"
with open(base_txt_file_path, "w") as fptr:
    fptr.write(json.dumps(all_face_features, indent=4))