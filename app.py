################# IMPORTS #############################
from flask import  Flask, request, jsonify, render_template,redirect, url_for, request,Response
import sys
#Debug logger
import logging 
import json
root = logging.getLogger()
root.setLevel(logging.DEBUG)



app = Flask(__name__,static_folder="./templates/assets")


########################################################### Codes 
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io





# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)




# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
global emotion
emotion = "Detecting Please Wait"
global data
data = {}
def gen():
    global emotion
    cap = cv2.VideoCapture(0)
    while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                emotion = emotion_dict[maxindex]

            #cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
            encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
            io_buf = io.BytesIO(image_buffer)
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

########################################################### Routes 
@app.route('/')
def hello_world():
    return render_template('splash.html')

global error
error = ""
@app.route('/log')
def log():
    global error
    return render_template('login.html', error=error)

global username
username = ""
@app.route('/register',methods=['POST','GET'])
def register():
    global username
    username = request.values.get("rg_username")
    password = request.values.get("rg_password")
    email = request.values.get("rg_email")
    list1 = [username,password,email]
    with open("users.csv", "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(list1)
    return redirect(url_for('home'))

@app.route('/home')
def helloworld():
    global emotion
    emotion = "Detecting Please Wait"
    return render_template('index.html',emotion=emotion)


@app.route('/admin')
def adminworld():
    return render_template('admin.html')


import csv
from collections import defaultdict
showwrongpassword = False
@app.route('/login',methods=['POST','GET'])
def logins():
    global error
    global username
    username = request.values.get("lg_username")
    password = request.values.get("lg_password")
    bol = False
    if(username == "admin" and password == "admin@123"):
        return redirect(url_for('adminworld'))
    with open("users.csv", 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if(row[0] == username and row[1] == password):
                error=""
                bol = True
        if bol:
            return redirect(url_for('helloworld'))
        else:
            showwrongpassword = True
            error ="Wrong Password, Please try login Again or Register"
            return redirect(url_for('log'))
    return


@app.route('/getuserdetails')
def getuserdetails():
    with open("users.csv", 'r') as csvfile:
        datareader = csv.reader(csvfile)
        list_user = []
        for row in datareader:
            list_user.append({"name":row[0],"email":row[2]})
    response = app.response_class(
        response=json.dumps(list_user),
        mimetype='application/json'
    )
    return response


@app.route('/getuserhistory')
def getuserhistory():
    with open("history.csv", 'r') as csvfile:
        datareader = csv.reader(csvfile)
        list_user = []
        for row in datareader:
            list_user.append({"name":row[0],"mood":row[1],"time":row[2]})
    response = app.response_class(
        response=json.dumps(list_user),
        mimetype='application/json'
    )
    return response

from pathlib import Path 
@app.route('/resultpage')
def resultpage():
    return render_template('result.html',emotion=emotion)

import datetime
@app.route('/getPlaylist')
def playlist():
    global emotion
    global username
    ct = datetime.datetime.now()
    list1 = [username,emotion,ct]
    with open("history.csv", "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(list1)
    list_songs = []
    DIRECTORY = Path('./templates/assets/'+emotion)
    for fp in DIRECTORY.glob('*.mp3'):
        value = str(fp).replace("templates\\", "")
        path = value.replace("\\","/")
        name = path.split("/")
        list_songs.append({"name":name[2],"note":"Playing a "+emotion+" song","file":path})
    response = app.response_class(
        response=json.dumps(list_songs),
        mimetype='application/json'
    )
    return response
    

@app.route('/getEmotion')
def get_emotion():
    global emotion
    res = {"emotion":emotion}
    response = app.response_class(
        response=json.dumps(res),
        mimetype='application/json'
    )
    return response



#launch a Tornado server with HTTPServer.
if __name__ == '__main__':
	app.run()
    
