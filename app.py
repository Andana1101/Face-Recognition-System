import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True

# Number of images to take for each user
nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Load the model once when the application starts
model_path = 'static/face_recognition_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    else:
        return []

# Identify face using ML model
def identify_face(facearray):
    if model is not None:
        return model.predict(facearray)
    return ["Unknown"]

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, model_path)
    global model
    model = knn

# Train the model at the start of the application if the model does not exist
if model is None:
    train_model()

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

################## ROUTING FUNCTIONS #######################
####### for Face Recognition based Attendance System #######

# Our main page
@app.route("/")
def home():
    return render_template('Home.html')

@app.route('/Attendance')
def attendance():
    names, rolls, times, l = extract_attendance()
    return render_template('Attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    if model is None:
        return render_template('Attendance.html', totalreg=totalreg(), mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)  # Make sure you use the correct camera index
    if not cap.isOpened():
        return render_template('Attendance.html', totalreg=totalreg(), mess='Unable to access the camera.')

    attended = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            if identified_person not in attended:
                add_attendance(identified_person)
                attended.add(identified_person)
            else:
                cv2.putText(frame, f'{identified_person}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27 or len(attended) >= 10:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('Attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)  # Make sure you use the correct camera index
        if not cap.isOpened():
            return render_template('Attendance.html', totalreg=totalreg(), mess='Unable to access the camera.')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs*5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        train_model()
        names, rolls, times, l = extract_attendance()
        return render_template('Attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())

    return render_template('Add.html', totalreg=totalreg())

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
