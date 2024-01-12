import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

ironman_image = face_recognition.load_image_file("ironman.jpeg")
ironman_encoding = face_recognition.face_encodings(ironman_image)[0]

captain_image = face_recognition.load_image_file("captainamerica.jpg")
captain_encoding = face_recognition.face_encodings(captain_image)[0]

spiderman_image = face_recognition.load_image_file("spiderman.jpg")
spiderman_encoding = face_recognition.face_encodings(spiderman_image)[0]

blackwidow_image = face_recognition.load_image_file("blackwidow.jpg")
blackwidow_encoding = face_recognition.face_encodings(blackwidow_image)[0]

thor_image = face_recognition.load_image_file("thor.jpg")
thor_encoding = face_recognition.face_encodings(thor_image)[0]

hulk_image = face_recognition.load_image_file("hulk.jpg")
hulk_encoding = face_recognition.face_encodings(hulk_image)[0]

doctorstrange_image = face_recognition.load_image_file("dr_Strange.jpeg")
doctorstrange_encoding = face_recognition.face_encodings(doctorstrange_image)[0]

blackpanther_image = face_recognition.load_image_file("blackpanther.jpg")
blackpanther_encoding = face_recognition.face_encodings(blackpanther_image)[0]

antman_image = face_recognition.load_image_file("antman.jpg")
antman_encoding = face_recognition.face_encodings(antman_image)[0]

scarletwitch_image = face_recognition.load_image_file("scarletwitch.jpg")
scarletwitch_encoding = face_recognition.face_encodings(scarletwitch_image)[0]

known_faces_encodings = [ironman_encoding, captain_encoding, spiderman_encoding,
                         blackwidow_encoding, thor_encoding, hulk_encoding,
                         doctorstrange_encoding, blackpanther_encoding,
                         antman_encoding, scarletwitch_encoding]

known_face_names = ["Iron Man", "Captain America", "Spider-Man", "Black Widow",
                    "Thor", "Hulk", "Doctor Strange", "Black Panther",
                    "Ant-Man", "Scarlet Witch"]

students = known_face_names.copy()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

with open(f"{current_date}.csv", "w+", newline="") as f:
    csv_writer = csv.writer(f)

    print("Facial Recognition System Initialized")

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            name = "Unknown"
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                font = cv2.FONT_HERSHEY_DUPLEX
                bottom_left_corner_of_text = (10, 80)
                font_scale = 0.8
                font_color = (0, 255, 0)  
                thickness = 1
                line_type = 2
                cv2.putText(frame, f"{name} Present", bottom_left_corner_of_text, font, font_scale, font_color,
                            thickness, line_type)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    csv_writer.writerow([name, current_time])
                    print(f"{name} Present")

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
