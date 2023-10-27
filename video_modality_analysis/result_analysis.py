import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import collections

json_path = "vid_frame_count.json"

# Opening JSON file
with open('vid_frame_count.json', "r") as fptr:
    json_data = json.load(fptr)

num_faces = []
for key in json_data.keys():
    for vals in json_data[key]:
        num_faces.append(vals["num_faces_each_frame"])
faces_freq = collections.Counter(num_faces)
plt.bar(faces_freq.keys(), faces_freq.values(), color="maroon", width=0.4)
plt.xlabel("Number of faces")
plt.ylabel("Occurence of these faces")
plt.title("Count of faces accross videos frames sampled @ 1 FPS")
plt.savefig("num_faces.png")
plt.show()

num_emotions = []
for key in json_data.keys():
    emotion = []
    for vals in json_data[key]:
        emotion.extend(vals["emotions"])
    num_emotions.append(collections.Counter(emotion).most_common(1)[0][0])
emottions_freq = collections.Counter(num_emotions)
plt.bar(emottions_freq.keys(), emottions_freq.values(), color="maroon", width=0.4)
plt.xlabel("Emotions")
plt.ylabel("Occurence of these emotions")
plt.title("Most dominant emotions accross each videos")
plt.savefig("emottions_freq.png")
plt.show()

num_ages = []
for key in json_data.keys():
    ages = []
    for vals in json_data[key]:
        ages.extend(vals["ages"])
    num_ages.append(collections.Counter(ages).most_common(1)[0][0])
ages_freq = collections.Counter(num_ages)
plt.bar(ages_freq.keys(), ages_freq.values(), color="maroon", width=0.4)
plt.xlabel("Age (with an error range of 5 years)")
plt.ylabel("Occurence of these ages")
plt.title("Most dominant ages accross each videos")
plt.savefig("ages_freq.png")
plt.show()

num_races = []
for key in json_data.keys():
    race = []
    for vals in json_data[key]:
        race.extend(vals["races"])
    num_races.append(collections.Counter(race).most_common(1)[0][0])
races_freq = collections.Counter(num_races)
plt.bar(races_freq.keys(), races_freq.values(), color="maroon", width=0.4)
plt.xlabel("Race")
plt.ylabel("Occurence of these race")
plt.title("Most dominant race accross each videos")
plt.savefig("races_freq.png")
plt.show()