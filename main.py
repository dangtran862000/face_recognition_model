import numpy as np
import cv2
import os
import face_recognition

import csv


def find_people(name):
    # csv file name
    filename = "demofile.txt"

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

    check = ""
    #  printing first 5 rows
    for row in rows:
        if row[0] == name:
            check = name
        if row[1] == "True":
            check = "True"
    if check == "True":
        return True
    else:
        return False


def initialize():
    path = 'Images'
    persons = os.listdir(path)

    encodes = []
    face_names = []
    i = 1
    for person in persons:
        files = os.listdir(path + '/' + person)
        for _ in files:
            image = face_recognition.load_image_file(path + '/' + person + '/' + str(i) + '.jpg')
            encod = face_recognition.face_encodings(image)[0]
            encodes.append(encod)
            break
        face_names.append(person)

    return face_names, encodes


face_names, encod_face = initialize()

locations = []
encodings = []


cap = cv2.VideoCapture(0)
i = 2
flag = True
names = ['Unknown']


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=1, fy=1)

    if i % 2 == 0:
        i = 0
        names = []

        locations = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, locations)

        for encoding in encodings:

            matches = face_recognition.compare_faces(encod_face, encoding)
            distance = face_recognition.face_distance(encod_face, encoding)
            best_matches = np.argmin(distance)

            if distance[best_matches] < 0.6:
                names.append(face_names[best_matches])
            else:
                names.append('Unknown')

    i += 1
    # cv2.imwrite("Images/Hai Dang/"+str(i)+".jpg",frame)

    for (top, right, bottom, left), name in zip(locations, names):
        if name == "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.rectangle(frame, (left, bottom - 17), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "Unknown", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        elif find_people(name) == False:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "NOT VACCINATED", (left + 6, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame,name , (left + 6, bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        elif find_people(name):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "VACCINATED", (left + 6, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, name, (left + 6, bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Testing", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

