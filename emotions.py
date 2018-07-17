import cv2
import numpy as np
import tkinter
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
import webbrowser
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import psycopg2

def OpenUrl(url):
    chromedir= 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    webbrowser.get(chromedir).open(url[0])

def feedback(pressed_button,num,name):
    print('Inside feedback')
    if pressed_button == "sad":
        print('Inside feedback1')
        print("UPDATE public.\"Music\" SET sad=sad+1 WHERE slink='{%s}' " % name)
        curr.execute("UPDATE public.\"Music\" SET sad=sad+1 WHERE slink='{%s}' " % name)

    if pressed_button == "happy":
        print('Inside feedback2')
        print("UPDATE public.\"Music\" SET happy=happy+1 WHERE slink='{%s}' " % name)
        curr.execute("UPDATE public.\"Music\" SET happy=happy+1 WHERE slink='{%s}' " % name)

    if pressed_button == "neutral":
        print('Inside feedback3')
        print("UPDATE public.\"Music\" SET neutral=neutral+1 WHERE slink='{%s}' " % name)
        curr.execute("UPDATE public.\"Music\" SET neutral=neutral+1 WHERE slink='{%s}' " % name)

    conn.commit()


def connection1(emotion):
    root=tkinter.Tk()
    global conn
    root.geometry("500x500")
    conn = psycopg2.connect(database="postgres", user = "postgres", password = "root", host = "127.0.0.1", port = "5432")
    if conn:
        print("Connectcion successful")
    global curr
    curr = conn.cursor()
    print(emotion)
#Getting the emotion here
#Either happy, sad or neutral
    curr.execute("SELECT * FROM public.\"Music\" ORDER BY "+emotion)
    cnt=1
    res = curr.fetchall()
    for row in res:
    #for i in range(4):
        l=tkinter.Label(root,text=row[4][0]).grid(row=cnt,column=1)
        b=tkinter.Button(root,text='Play',command=lambda: OpenUrl(row[0])).grid(row=cnt,column=2)
        b=tkinter.Button(root,text='Happy',command=lambda: feedback('happy',row[1],row[0][0])).grid(row=cnt,column=3)
        b=tkinter.Button(root,text='Sad',command=lambda: feedback('sad',row[2],row[0][0])).grid(row=cnt,column=4)
        b=tkinter.Button(root,text='Nuetral',command=lambda: feedback('nuetral',row[3],row[0][0])).grid(row=cnt,column=5)
        cnt+=1
        print("link: ",row[0])
        print("happy:",row[1])
        print("sad:",row[2])
        print("neutral:",row[3])
        print("name:",  row[4])
    root.mainloop()


    

USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        print('c pressed')
        '''if(emotion_text=='sad'):
            emotion_text = emotion_text +"You seem to be in a sad mood....."
            emotion_text = emotion_text +"Here is a playlist that might help..."
            global w
            #w = tkinter.Label(root, text=emotion_text)
            #w.pack()
            b=tkinter.Button(root,text='Play',command=OpenUrl("https://www.youtube.com/watch?v=Sv6dMFF_yts&list=PLvPjBbFpr-O1azdoLQpxSt9nX5twB3kky"))
            b.pack()
            root.mainloop()
        elif(emotion_text=='happy'):
            emotion_text = emotion_text +"You seem to be in a great mood. This can further cheer you up!!!!"
            global w
            #w = tkinter.Label(root, text=emotion_text)
            #w.pack()
            b=tkinter.Button(root,text='Play',command=OpenUrl("https://www.youtube.com/watch?v=LjhCEhWiKXk&list=PL1VuYyZcPYIJTP3W_x0jq9olXviPQlOe1"))
            b.pack()
            root.mainloop()
        else:
            emotion_text=emotion_text+"We still have something for you !!"
            global w
            b=tkinter.Button(root,text='Play',command=OpenUrl("https://www.youtube.com/watch?v=CcsUYu0PVxY&start_radio=1&list=RDQMfmSBrybf-ro"))
    '''
        connection1(emotion_text)
        
cap.release()
cv2.destroyAllWindows()
