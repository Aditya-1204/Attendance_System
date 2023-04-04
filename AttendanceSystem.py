import cv2
import numpy as np
import os
import csv
import datetime
import tkinter as tk
from tkinter import filedialog

# Initialize GUI
gui = tk.Tk()
gui.title("SMART ATTENDANCE SYSTEM USING OPENCV")

# Initialize face detection model
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize attendance csv file
header = ['Roll No', 'Name', 'Date', 'Time']
csv_file = open('attendance.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(header)

# Initialize student database
student_database = {}

# Function to add a student to the database
def add_student_to_database(name):
    roll_no = len(student_database) + 1
    student_database[roll_no] = name
    return roll_no

# Function to train the face recognition model with a student's face
def train_face():
    name = name_entry.get()
    roll_no = add_student_to_database(name)
    
    # Capture a live image from the camera
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    cv2.imwrite("dataset/face_" + str(roll_no) + ".jpg", image)
    camera.release()

# Function to start the face detection and recognition process
def start_recognition():
    # Open the camera
    camera = cv2.VideoCapture(0)
    
    while True:
        # Capture a frame from the camera
        return_value, frame = camera.read()
        if not return_value:
            break
        
        # Convert the frame to grayscale for better accuracy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Recognize the face using the face recognition model
            face_image = gray[y:y+h, x:x+w]
            roll_no, confidence = recognize_face(face_image)
            if roll_no is not None and confidence < 100:
                name = student_database[roll_no]
                
                # Mark attendance in the csv file
                now = datetime.datetime.now()
                date_time = now.strftime("%Y-%m-%d %H:%M:%S")
                row = [roll_no, name, date_time.split()[0], date_time.split()[1]]
                csv_writer.writerow(row)
        
        # Display the resulting frame
        cv2.imshow('Attendance System', frame)
        
        # Press 'q' to stop the camera and attendance marking
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the camera and close the GUI
    camera.release()
    csv_file.close()
    cv2.destroyAllWindows()

# Function to recognize a face using the face recognition model
def recognize_face(face_image):
    # Load the saved face recognition model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    
    # Recognize the face in the image
    roll_no, confidence = recognizer.predict(face_image)
    if confidence > 100:
        return None, confidence
    else:
        return
