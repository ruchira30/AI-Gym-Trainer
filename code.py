import cv2
import mediapipe as mp
from PIL import Image, ImageTk
from os import listdir, chdir
from matplotlib import pyplot as plt
from tkinter import Tk, Toplevel, ttk, filedialog, CENTER, Button, Frame, Canvas, Label, Scrollbar, LEFT, RIGHT, BOTH, X, Y, HORIZONTAL, Scale, Entry, BOTTOM
import numpy as np
from tqdm import tqdm
from pygame import init, mixer, USEREVENT, event
from keyboard import add_hotkey
from threading import Thread
from random import shuffle
from time import sleep
from ctypes import windll
import pyttsx3
from functools import partial
from moviepy.audio.io.AudioFileClip import AudioFileClip as audio

mp_drawing = mp.solutions.drawing_utils
windll.shcore.SetProcessDpiAwareness(True)

global voice
voice = 0
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) # only 0,1
engine.setProperty('volume',1.0) #0.0 to 1.0
engine.setProperty('rate', 175) #words per minute

def speak(audio):
    if engine._inLoop:
        pass
    else:
        speak_audio = Thread(target = say_audio, daemon = True, args = (audio,))
        speak_audio.start()

def say_audio(audio):
    #global vol
    orignalvol = vol
    w.set(20)
    engine.say(audio)
    engine.runAndWait()
    w.set(orignalvol)

def show_image(img, save='', figsize=(10, 10)):
    """Shows output PIL image."""

    if save=='save':
        cv2.imwrite('Output_generated.png', img)
    else:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()

mp_pose = mp.solutions.pose

segments = ['LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER', 'LEFT_EYE_OUTER', 'LEFT_FOOT_INDEX',
            'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX',
            'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_WRIST', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE',
            'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_ELBOW',
            'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP',
            'RIGHT_INDEX', 'RIGHT_KNEE', 'RIGHT_PINKY',
            'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST']

def get_points(img, view=''):
    image = cv2.imread(img)

    image_height, image_width, _ = image.shape
    points = list()
    finalimage = image
    if view == 'skel':
        finalimage = np.zeros(shape=[round(image_height), round(image_width), round(_)], dtype=np.uint8)

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for idx, segment in enumerate(segments):
            try:
                exec('global x;x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment + '].x * image_width')
                exec('global y;y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment + '].y * image_height')
                # print(idx, segment, x, y)
            except:
                pass
            points.append((round(x),round(y)))
            radius = 1
            center = (int(x), int(y))
            color = (255, 255, 0)
            thickness = 2
            image = cv2.circle(finalimage, center, radius, color, thickness)

    return points, finalimage

def embed(image, view=''):
    image = cv2.imread(image)
    image_height, image_width, _ = image.shape
    finalimage = image
    if view == 'skel':
        finalimage = np.zeros(shape=[round(image_height), round(image_width), round(_)], dtype=np.uint8)
    lines = (('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP'), ('LEFT_ANKLE', 'LEFT_FOOT_INDEX'),
             ('LEFT_ANKLE', 'LEFT_KNEE'),
             ('LEFT_KNEE', 'LEFT_HIP'), ('LEFT_HIP', 'LEFT_SHOULDER'), ('LEFT_SHOULDER', 'LEFT_ELBOW'),
             ('LEFT_ELBOW', 'LEFT_WRIST'),
             ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'), ('RIGHT_ANKLE', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_HIP'),
             ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'))
    distance = list()
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i in lines:
            segment1 = i[0]
            segment2 = i[1]
            exec(
                'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * image_width)')
            exec(
                'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * image_height)')
            exec(
                'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * image_width)')
            exec(
                'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * image_height)')
            cv2.line(finalimage, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
            dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist = round(dist ** 0.5)
            distance.append(dist)

        return distance, finalimage

def embed_video(vid, view=''):
    cap = cv2.VideoCapture(vid)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    height, width, layers = frame.shape
    size = (width, height)

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    lines = (('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP'), ('LEFT_ANKLE', 'LEFT_FOOT_INDEX'),
             ('LEFT_ANKLE', 'LEFT_KNEE'),
             ('LEFT_KNEE', 'LEFT_HIP'), ('LEFT_HIP', 'LEFT_SHOULDER'), ('LEFT_SHOULDER', 'LEFT_ELBOW'),
             ('LEFT_ELBOW', 'LEFT_WRIST'),
             ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'), ('RIGHT_ANKLE', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_HIP'),
             ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'))

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
    ) as pose:
        for count in tqdm(range(count)):
            ret, frame = cap.read()

            try:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if view == 'skel':
                    frame = np.zeros(shape=[round(height), round(width), round(layers)], dtype=np.uint8)
            except:
                continue

            for i in lines:
                segment1 = i[0]
                segment2 = i[1]

                try:
                    exec(
                        'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * width)')
                    exec(
                        'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * height)')
                    exec(
                        'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * width)')
                    exec(
                        'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * height)')
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)

                except:
                    print('Error in frame : ',count)
                    break

            out.write(frame)

    out.release()
    cap.release()

def embed_style(image, view=''):
    image = cv2.imread(image)
    image_height, image_width, _ = image.shape
    finalimage = image

    segments = ('LEFT_ANKLE', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_HIP',
                'LEFT_KNEE', 'LEFT_SHOULDER',
                'RIGHT_ANKLE', 'RIGHT_ELBOW',
                'RIGHT_EYE', 'RIGHT_HIP',
                'RIGHT_KNEE',
                'RIGHT_SHOULDER')

    lines_left = (('LEFT_ANKLE', 'LEFT_FOOT_INDEX'),('LEFT_ANKLE','LEFT_KNEE'),
            ('LEFT_KNEE','LEFT_HIP'),('LEFT_HIP','LEFT_SHOULDER'),('LEFT_SHOULDER','LEFT_ELBOW'),('LEFT_ELBOW','LEFT_WRIST'),
            ('LEFT_HEEL','LEFT_ANKLE'),('LEFT_HEEL','LEFT_FOOT_INDEX'))

    lines_right = (('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'),('RIGHT_ANKLE','RIGHT_KNEE'),('RIGHT_KNEE','RIGHT_HIP'),
            ('RIGHT_HIP','RIGHT_SHOULDER'),('RIGHT_SHOULDER','RIGHT_ELBOW'),('RIGHT_ELBOW','RIGHT_WRIST'),
            ('RIGHT_HEEL','RIGHT_ANKLE'),('RIGHT_HEEL','RIGHT_FOOT_INDEX'))

    lines_center = (('LEFT_SHOULDER', 'RIGHT_SHOULDER'),('LEFT_HIP','RIGHT_HIP'),('MOUTH_LEFT', 'MOUTH_RIGHT'))

    if view=='skel':
        finalimage = np.zeros(shape=[round(image_height), round(image_width), round(_)], dtype=np.uint8)

    distance = list()
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for segment in segments:
            try:
                exec('global x;x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment + '].x * image_width')
                exec('global y;y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment + '].y * image_height')
            except:
                pass
            radius = 3
            center = (int(x), int(y))
            color = (255, 200, 150)
            thickness = 2
            image = cv2.circle(finalimage, center, radius, color, thickness)

        for i in lines_left:
            segment1 = i[0]
            segment2 = i[1]
            exec('global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * image_width)')
            exec('global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * image_height)')
            exec('global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * image_width)')
            exec('global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * image_height)')
            cv2.line(finalimage, (x1, y1), (x2, y2), (255, 150, 155), thickness=2)

        for i in lines_right:
            segment1 = i[0]
            segment2 = i[1]
            exec('global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * image_width)')
            exec('global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * image_height)')
            exec('global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * image_width)')
            exec('global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * image_height)')
            cv2.line(finalimage, (x1, y1), (x2, y2), (155, 255, 155), thickness=2)

        for i in lines_center:
            segment1 = i[0]
            segment2 = i[1]
            exec('global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * image_width)')
            exec('global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * image_height)')
            exec('global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * image_width)')
            exec('global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * image_height)')
            cv2.line(finalimage, (x1, y1), (x2, y2), (220, 220, 220), thickness=2)

        return finalimage

def embed_video_style(vid, view=''):
    cap = cv2.VideoCapture(vid)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    height, width, layers = frame.shape
    size = (width, height)

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    segments = ('LEFT_ANKLE', 'LEFT_ELBOW', 'LEFT_HIP',
                'LEFT_KNEE', 'LEFT_SHOULDER',
                'RIGHT_ANKLE', 'RIGHT_ELBOW', 'RIGHT_HIP',
                #'RIGHT_EYE', 'LEFT_EYE',
                'RIGHT_KNEE',
                'RIGHT_SHOULDER')

    lines_left = (('LEFT_ANKLE', 'LEFT_FOOT_INDEX'), ('LEFT_ANKLE', 'LEFT_KNEE'),
                  ('LEFT_KNEE', 'LEFT_HIP'), ('LEFT_HIP', 'LEFT_SHOULDER'), ('LEFT_SHOULDER', 'LEFT_ELBOW'),
                  ('LEFT_ELBOW', 'LEFT_WRIST'),
                  ('LEFT_HEEL', 'LEFT_ANKLE'), ('LEFT_HEEL', 'LEFT_FOOT_INDEX'))

    lines_right = (('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'), ('RIGHT_ANKLE', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_HIP'),
                   ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
                   ('RIGHT_HEEL', 'RIGHT_ANKLE'), ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX'))

    lines_center = (('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP'),
                    #('MOUTH_LEFT', 'MOUTH_RIGHT')
                    )

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
    ) as pose:
        for count in tqdm(range(count)):
            ret, frame = cap.read()

            try:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                #results = pose.process(frame)
                if view == 'skel':
                    frame = np.zeros(shape=[round(height), round(width), round(layers)], dtype=np.uint8)
            except:
                continue

            try:
                for segment in segments:
                    exec(
                            'global x;x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment + '].x * width')
                    exec(
                            'global y;y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment + '].y * height')
                    radius = 3
                    center = (int(x), int(y))
                    color = (255, 200, 150)
                    thickness = 2
                    image = cv2.circle(frame, center, radius, color, thickness)

                for i in lines_left:
                    segment1 = i[0]
                    segment2 = i[1]
                    exec(
                        'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * width)')
                    exec(
                        'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * height)')
                    exec(
                        'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * width)')
                    exec(
                        'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * height)')
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 155), thickness=2)

                for i in lines_right:
                    segment1 = i[0]
                    segment2 = i[1]
                    exec(
                        'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * width)')
                    exec(
                        'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * height)')
                    exec(
                        'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * width)')
                    exec(
                        'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * height)')
                    cv2.line(frame, (x1, y1), (x2, y2), (155, 255, 155), thickness=2)

                for i in lines_center:
                    segment1 = i[0]
                    segment2 = i[1]
                    exec(
                        'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * width)')
                    exec(
                        'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * height)')
                    exec(
                        'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * width)')
                    exec(
                        'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * height)')
                    cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), thickness=2)

                out.write(frame)

            except Exception as e:
                continue

    out.release()
    cap.release()

def get_dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dist = (x1 - x2) ** 2 + (y1 - y2) ** 2
    dist = (dist ** 0.5)
    return(dist)

def get_angle_x(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    hypo = (x1 - x2) ** 2 + (y1 - y2) ** 2
    hypo = (hypo ** 0.5)

    base = abs(x1-x2)
    
    ang = np.arccos(base/hypo)
    ang = ang*180/np.pi
    return(ang)

def get_angle_y(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    hypo = (x1 - x2) ** 2 + (y1 - y2) ** 2
    hypo = (hypo ** 0.5)

    per = abs(y1-y2)
    
    ang = np.arccos(per/hypo)
    ang = ang*180/np.pi
    return(ang)

def add_frames(text, frames, shape, output):
    height, width, layers = shape
    for i in range(frames):
        frame = np.zeros(shape=[round(height), round(width), round(layers)], dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        frame = cv2.putText(frame, text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

        output.write(frame)

def dynamic_workout_upload(limiti, segments, deciding_comparisions, deciding_angles, angles, deductions, feedback, comparisions, anglesi, deductionsi, feedbacki, comparisionsi, angles2, deductions2, feedback2, comparisions2, deciding_comparisions_lowest, deciding_comparisions_highest, angles_lowest, deductions_lowest, feedback_lowest, comparisions_lowest, angles_highest, deductions_highest, feedback_highest, comparisions_highest, checking_comparisions):
    global current_workout
    status_label.config(text='', foreground='#d0d0d0')
    fps = camera.get(cv2.CAP_PROP_FPS)
    ret, frame = camera.read()
    height, width, layers = frame.shape
    size = (width, height)
    download_file = filedialog.asksaveasfilename(filetypes=(("Video", "*.mp4"),))
    if download_file == '':
        download_file = 'C:\\Users\\vaibh\\Desktop\\Exercise form\\Advance report.mp4'
    else:
        download_file += '.mp4'
    out = cv2.VideoWriter(download_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    start = 0
    nt = 0
    time = 0
    time_rep = 1
    message = ''
    len_to_compare = len(deciding_comparisions)
    scores = list()
    score = 0
    prev_pos = 0
    reps = 0
    time_frames = 0
    final_message = ''
    final_ded = 0
    lowestdist = 0
    highestdist = 0
    highest_frame = ''
    lowest_frame = ''
    frames_add = 1 / fps
    message_highest = ''
    message_lowest = ''
    len_to_compare = len(deciding_comparisions)
    # aim = 60 # seconds
    add_frames('Vitual Gym Trainer', int(fps * 2), frame.shape, out)
    add_frames('Workout : ' + current_workout, int(fps * 2), frame.shape, out)
    add_frames("Get in the correct pose to start the workout", 60, frame.shape, out)

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
    ) as pose:
        for st in range(count):
            ret, frame = camera.read()
            try:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            except Exception as e:
                print(e)
                continue

            try:
                for i in segments:
                    exec(
                        'global x' + i + '; x' + i + '= results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + i + '].x * width')
                    exec(
                        'global y' + i + '; y' + i + '= results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + i + '].y * height')

                ded = 0
                position = 0

                exec('global ch; ch = get_dist((x' + checking_comparisions[0] + ', y' + checking_comparisions[
                    0] + '),(x' + checking_comparisions[1] + ', y' + checking_comparisions[1] + '))')
                exec('global chc; chc = get_dist((x' + checking_comparisions[2] + ', y' + checking_comparisions[
                    2] + '),(x' + checking_comparisions[3] + ', y' + checking_comparisions[3] + '))')
                
                if chc > ch:
                    print('Out of pose')
                    continue

                for n, i in enumerate(deciding_comparisions):
                    if i[2] == 'x':
                        exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                            1] + '))')

                        if ang < deciding_angles[n][0]:
                            pass

                        elif ang > deciding_angles[n][1]:
                            position += 1

                        else:
                            position += 0.5

                if position == 0:
                    position = 'Low'
                    if prev_pos == len_to_compare:
                        prev_pos = 0
                        reps += 1
                        lowestdist = 0

                    message = ''
                    for n, i in enumerate(comparisions):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                message += feedback[n]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]

                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                message += feedback[n]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]

                    score = 100 - ded
                    if ded <= final_ded:
                        final_ded = ded
                        scores.append(score)

                    time += frames_add

                    for n, i in enumerate(deciding_comparisions_lowest):
                        exec('global lowdist; lowdist = get_dist((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                            1] + '))')
                        if lowdist > lowestdist:
                            lowestdist = lowdist
                            for n, i in enumerate(comparisions_lowest):
                                if i[2] == 'x':
                                    exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[
                                        1] + ', y' + i[1] + '))')
                                    if ang > angles_lowest[n][0]:
                                        message_highest += feedback_lowest[n]

                                if i[2] == 'y':
                                    exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[
                                        1] + ', y' + i[1] + '))')
                                    if ang > angles_lowest[n][0]:
                                        message_highest += feedback_lowest[n]

                elif position == len_to_compare:
                    position = 'High'
                    if prev_pos == 0:
                        prev_pos = len_to_compare
                        highestdist = 0

                    message = ''
                    for n, i in enumerate(comparisions2):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > angles2[n][0]:
                                ded += deductions2[n][0]
                                message += feedback2[n]
                                if ang > angles2[n][1]:
                                    ded += deductions2[n][1]

                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > angles2[n][0]:
                                ded += deductions2[n][0]
                                message += feedback2[n]
                                if ang > angles2[n][1]:
                                    ded += deductions2[n][1]

                    score = 100 - ded
                    if ded <= final_ded:
                        final_ded = ded
                        scores.append(score)

                    time += frames_add

                    for n, i in enumerate(deciding_comparisions_highest):
                        exec(
                            'global highdist; highdist = get_dist((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                        if highdist > highestdist:
                            highestdist = highdist
                            for n, i in enumerate(comparisions_highest):
                                if i[2] == 'x':
                                    exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[
                                        1] + ', y' + i[1] + '))')
                                    if ang > angles_highest[n][0]:
                                        message_highest += feedback_highest[n]

                                if i[2] == 'y':
                                    exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[
                                        1] + ', y' + i[1] + '))')
                                    if ang > angles_highest[n][0]:
                                        message_highest += feedback_highest[n]

                else:
                    message = ''
                    position = 'Intermediate'
                    if message_highest != '':
                        message += message_highest
                        message_highest = ''

                    elif message_lowest != '':
                        message += message_lowest
                        message_lowest = ''

                    for n, i in enumerate(comparisionsi):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > anglesi[n][0]:
                                ded += deductionsi[n][0]
                                message += feedbacki[n]
                                if ang > anglesi[n][1]:
                                    ded += deductionsi[n][1]

                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > anglesi[n][0]:
                                ded += deductionsi[n][0]
                                message += feedbacki[n]
                                if ang > anglesi[n][1]:
                                    ded += deductionsi[n][1]

                    score = 100 - ded
                    if ded <= limiti:
                        time += frames_add

                    else:
                        message = 'Out of pose'

            except Exception as e:
                # print('Could not apply segmentation to frame')
                print(e)
                continue

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            color2 = (0, 255, 0)
            org2 = 50, 200
            org3 = 400, 200
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if message == '':
                frame = cv2.putText(frame, "Correct", org, font, fontScale, (100, 255, 100), thickness, cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, "Message : " + message, org, font, fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, "Time : " + str(time), org2, font, fontScale, color2, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, "Position : " + position, org3, font, fontScale, color, thickness,cv2.LINE_AA)

            out.write(frame)
            percdone = st / count * 100
            hp_bar2['value'] = percdone
            status_label.config(text=str(percdone) + ' % complete')

    hp_bar2['value'] = 100
    status_label.config(text='Task Complete', foreground='green')
    camera.release()
    out.release()
    return ()

def dynamic_workout_live(limiti, segments, deciding_comparisions, deciding_angles, angles, deductions, feedback, comparisions, anglesi, deductionsi, feedbacki, comparisionsi, angles2, deductions2, feedback2, comparisions2, deciding_comparisions_lowest, deciding_comparisions_highest, angles_lowest, deductions_lowest, feedback_lowest, comparisions_lowest, angles_highest, deductions_highest, feedback_highest, comparisions_highest, checking_comparisions):
    fps = camera.get(cv2.CAP_PROP_FPS)
    ret, frame = camera.read()
    height, width, layers = frame.shape
    size = (width, height)
    out = cv2.VideoWriter('C:\\Users\\vaibh\\Desktop\\Exercise form\\out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          size)
    start = 0
    nt = 0
    time = 0
    time_rep = 1
    message = ''
    len_to_compare = len(deciding_comparisions)
    scores = list()
    score = 0
    prev_pos = 0
    reps = 0
    time_frames = 0
    final_message = ''
    final_ded = 0
    lowestdist = 0
    highestdist = 0
    highest_frame = ''
    lowest_frame = ''
    frames_add = 1 / fps
    message_highest = ''
    message_lowest = ''
    len_to_compare = len(deciding_comparisions)
    speak("Get in the correct pose to start the workout")

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
    ) as pose:
        while True:
            global run
            ret, frame = camera.read()
            out.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = pose.process(frame)

            except:
                continue

            try:
                for i in segments:
                    exec(
                        'global x' + i + '; x' + i + '= results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + i + '].x * width')
                    exec(
                        'global y' + i + '; y' + i + '= results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + i + '].y * height')

                ded = 0
                position = 0

                exec('global ch; ch = get_dist((x' + checking_comparisions[0] + ', y' + checking_comparisions[
                    0] + '),(x' + checking_comparisions[1] + ', y' + checking_comparisions[1] + '))')
                exec('global chc; chc = get_dist((x' + checking_comparisions[2] + ', y' + checking_comparisions[2] + '),(x' + checking_comparisions[3] + ', y' + checking_comparisions[3] + '))')

                if chc > ch:
                    print('Out of pose')
                    continue

                for n, i in enumerate(deciding_comparisions):
                    if i[2] == 'x':
                        exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                            1] + '))')

                        if ang < deciding_angles[n][0]:
                            pass

                        elif ang > deciding_angles[n][1]:
                            position += 1

                        else:
                            position += 0.5

                if position == 0:
                    position = 'Low'
                    if prev_pos == len_to_compare:
                        prev_pos = 0
                        reps += 1
                        lowestdist = 0

                    message = ''
                    for n, i in enumerate(comparisions):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                message += feedback[n]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]

                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                message += feedback[n]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]

                    score = 100 - ded
                    if ded <= final_ded:
                        final_ded = ded
                        scores.append(score)

                    time += frames_add

                    for n, i in enumerate(deciding_comparisions_lowest):
                        exec('global lowdist; lowdist = get_dist((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                            1] + '))')
                        if lowdist > lowestdist:
                            lowestdist = lowdist
                            for n, i in enumerate(comparisions_lowest):
                                if i[2] == 'x':
                                    exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[
                                        1] + ', y' + i[1] + '))')
                                    if ang > angles_lowest[n][0]:
                                        message_highest += feedback_lowest[n]

                                if i[2] == 'y':
                                    exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[
                                        1] + ', y' + i[1] + '))')
                                    if ang > angles_lowest[n][0]:
                                        message_highest += feedback_lowest[n]

                elif position == len_to_compare:
                    position = 'High'
                    if prev_pos == 0:
                        prev_pos = len_to_compare
                        highestdist = 0

                    message = ''
                    for n, i in enumerate(comparisions2):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > angles2[n][0]:
                                ded += deductions2[n][0]
                                message += feedback2[n]
                                if ang > angles2[n][1]:
                                    ded += deductions2[n][1]

                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > angles2[n][0]:
                                ded += deductions2[n][0]
                                message += feedback2[n]
                                if ang > angles2[n][1]:
                                    ded += deductions2[n][1]

                    score = 100 - ded
                    if ded <= final_ded:
                        final_ded = ded
                        scores.append(score)

                    time += frames_add

                    for n, i in enumerate(deciding_comparisions_highest):
                        exec(
                            'global highdist; highdist = get_dist((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                        if highdist > highestdist:
                            highestdist = highdist
                            for n, i in enumerate(comparisions_highest):
                                if i[2] == 'x':
                                    exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[
                                        1] + ', y' + i[1] + '))')
                                    if ang > angles_highest[n][0]:
                                        message_highest += feedback_highest[n]

                                if i[2] == 'y':
                                    exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[
                                        1] + ', y' + i[1] + '))')
                                    if ang > angles_highest[n][0]:
                                        message_highest += feedback_highest[n]

                else:
                    message = ''
                    position = 'Intermediate'
                    if message_highest != '':
                        message += message_highest
                        message_highest = ''

                    elif message_lowest != '':
                        message += message_lowest
                        message_lowest = ''

                    for n, i in enumerate(comparisionsi):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > anglesi[n][0]:
                                ded += deductionsi[n][0]
                                message += feedbacki[n]
                                if ang > anglesi[n][1]:
                                    ded += deductionsi[n][1]

                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x' + i[0] + ', y' + i[0] + '),(x' + i[1] + ', y' + i[
                                1] + '))')
                            if ang > anglesi[n][0]:
                                ded += deductionsi[n][0]
                                message += feedbacki[n]
                                if ang > anglesi[n][1]:
                                    ded += deductionsi[n][1]

                    score = 100 - ded
                    if ded <= limiti:
                        time += frames_add

                    else:
                        message = 'Out of pose'

            except Exception as e:
                # print('Could not apply segmentation to frame')
                print(e)
                continue

            if time > nt:
                if message != '':
                    speak(message)
                    nt = time + 3

            time_label.config(text='Time : ' + str(time))
            message_label.config(text='Message : ' + message)
            reps_label.config(text='Reps : '+ str(reps))
            # calories_label.config(text = )

            if run == 1:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=img)
                label_image.configure(image=photo)
                label_image.photo = photo
                pass

            else:
                camera.release()
                out.release()
                break

    return ()

def static_workout_upload(limits, segments, angles, deductions, feedback, comparisions):
    global current_workout
    status_label.config(text='', foreground='#d0d0d0')
    fps = camera.get(cv2.CAP_PROP_FPS)
    to_add = 1/fps
    ret, frame = camera.read()
    height, width, layers = frame.shape
    size = (width, height)
    download_file = filedialog.asksaveasfilename(filetypes=(("Video", "*.mp4"),))
    if download_file == '':
        download_file = 'C:\\Users\\vaibh\\Desktop\\Exercise form\\Advance report.mp4'
    else:
        download_file += '.mp4'
    out = cv2.VideoWriter(download_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    start = 0
    nt = 0
    time = 0
    limit1, limit2 = limits
    score = 0
    message = ''
    #aim = 60 # seconds
    add_frames('Vitual Gym Trainer', int(fps*2), frame.shape, out)
    add_frames('Workout : '+ current_workout, int(fps*2), frame.shape, out)
    add_frames("Get in the correct pose to start the workout", 60, frame.shape, out)
    
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
    ) as pose:
        for st in range(count):
            ret, frame = camera.read()
            try:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
            except Exception as e:
                #print(e)
                continue

            try:
                for i in segments:
                    exec('global x'+i+'; x'+ i +'= results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + i + '].x * width')
                    exec('global y'+i+'; y'+ i +'= results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + i + '].y * height')

                ded = 0
                
                if start == 0:
                    for n, i in enumerate(comparisions):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x'+i[0]+', y'+i[0]+'),(x'+i[1]+', y'+i[1]+'))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]
                            
                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x'+i[0]+', y'+i[0]+'),(x'+i[1]+', y'+i[1]+'))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]
                            
                    if ded <= limit1:
                        start = 1
                        score = 100 - ded
                        time += to_add
                        message = "Workout started"

                if start == 1:
                    message = ''
                    for n, i in enumerate(comparisions):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x'+i[0]+', y'+i[0]+'),(x'+i[1]+', y'+i[1]+'))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                message += feedback[n]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]
                            
                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x'+i[0]+', y'+i[0]+'),(x'+i[1]+', y'+i[1]+'))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                message += feedback[n]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]

                    if ded <= limit2:
                        score = 100-ded
                        time += to_add

                    else:
                        start = 0
                        message = ("workout stopped")
                        
            except Exception as e:
                #print('Could not apply segmentation to frame')
                #print(e)
                continue

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            color2 = (0, 255 ,0)
            org2 = 50, 200
            org3 = 400, 200

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if message == '':
                frame = cv2.putText(frame, "Message : " + message, org, font, fontScale, (100, 255, 100), thickness, cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, "Message : " + message, org, font, fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, "Time : " + str(time), org2, font, fontScale, color2, thickness, cv2.LINE_AA)
            
            out.write(frame)
            percdone = st/count*100
            hp_bar2['value'] = percdone
            status_label.config(text=str(percdone)+' % complete')

    hp_bar2['value'] = 100
    status_label.config(text='Task Complete', foreground='green')
    camera.release()
    out.release()
    return()

def static_workout_live(limits, segments, angles, deductions, feedback, comparisions):

    fps = camera.get(cv2.CAP_PROP_FPS)
    to_add = 1/fps
    ret, frame = camera.read()
    height, width, layers = frame.shape
    size = (width, height)
    out = cv2.VideoWriter('C:\\Users\\vaibh\\Desktop\\Exercise form\\out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    start = 0
    nt = 0
    time = 0
    limit1, limit2 = limits
    score = 0
    message = ''
    #aim = 60 # seconds
    speak("Get in the correct pose to start the workout")
    
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
    ) as pose:
        while True:
            global run
            ret, frame = camera.read()
            out.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = pose.process(frame)
                
            except:
                continue

            try:
                for i in segments:
                    exec('global x'+i+'; x'+ i +'= results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + i + '].x * width')
                    exec('global y'+i+'; y'+ i +'= results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + i + '].y * height')

                ded = 0
                
                if start == 0:

                    for n, i in enumerate(comparisions):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x'+i[0]+', y'+i[0]+'),(x'+i[1]+', y'+i[1]+'))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]
                            
                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x'+i[0]+', y'+i[0]+'),(x'+i[1]+', y'+i[1]+'))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]
                            
                    if ded <= limit1:
                        start = 1
                        score = 100 - ded
                        time += to_add
                        #speak("Workout started")
                        #nt = time+3

                if start == 1:
                    message = ''
                    for n, i in enumerate(comparisions):
                        if i[2] == 'x':
                            exec('global ang; ang = get_angle_x((x'+i[0]+', y'+i[0]+'),(x'+i[1]+', y'+i[1]+'))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                message += feedback[n]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]
                            
                        if i[2] == 'y':
                            exec('global ang; ang = get_angle_y((x'+i[0]+', y'+i[0]+'),(x'+i[1]+', y'+i[1]+'))')
                            if ang > angles[n][0]:
                                ded += deductions[n][0]
                                message += feedback[n]
                                if ang > angles[n][1]:
                                    ded += deductions[n][1]

                    if ded <= limit2:
                        score = 100-ded
                        time += to_add

                    else:
                        start = 0
                        #speak("workout stopped")
                        #nt = time+3
                        
            except Exception as e:
                #print('Could not apply segmentation to frame')
                continue

            if time>nt:
                if message != '':
                    speak(message)
                    nt = time+3

            time_label.config(text = 'Time : '+str(time))
            message_label.config(text = 'Message : '+ message)
            # calories_label.config(text = )

            if run == 1:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=img)
                label_image.configure(image=photo)
                label_image.photo = photo
                pass

            else:
                camera.release()
                out.release()
                break

    return()

def embed_video_style_live():
    count = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = camera.read()
    height, width, layers = frame.shape
    size = (width, height)

    segments = ('LEFT_ANKLE', 'LEFT_ELBOW', 'LEFT_HIP',
                'LEFT_KNEE', 'LEFT_SHOULDER',
                'RIGHT_ANKLE', 'RIGHT_ELBOW', 'RIGHT_HIP',
                #'RIGHT_EYE', 'LEFT_EYE',
                'RIGHT_KNEE',
                'RIGHT_SHOULDER')

    lines_left = (('LEFT_ANKLE', 'LEFT_FOOT_INDEX'), ('LEFT_ANKLE', 'LEFT_KNEE'),
                  ('LEFT_KNEE', 'LEFT_HIP'), ('LEFT_HIP', 'LEFT_SHOULDER'), ('LEFT_SHOULDER', 'LEFT_ELBOW'),
                  ('LEFT_ELBOW', 'LEFT_WRIST'),
                  ('LEFT_HEEL', 'LEFT_ANKLE'), ('LEFT_HEEL', 'LEFT_FOOT_INDEX'))

    lines_right = (('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'), ('RIGHT_ANKLE', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_HIP'),
                   ('RIGHT_HIP', 'RIGHT_SHOULDER'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
                   ('RIGHT_HEEL', 'RIGHT_ANKLE'), ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX'))

    lines_center = (('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP'),
                    #('MOUTH_LEFT', 'MOUTH_RIGHT')
                    )

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
    ) as pose:
        while True:
            ret, frame = camera.read()

            try:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                #results = pose.process(frame)
                frame = np.zeros(shape=[round(height), round(width), round(layers)], dtype=np.uint8)
            except:
                continue

            try:
                # exec(
                #     'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + 'LEFT_SHOULDER' + '].x * width)')
                # exec(
                #     'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + 'LEFT_SHOULDER' + '].y * height)')
                # exec(
                #     'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + 'RIGHT_SHOULDER' + '].x * width)')
                # exec(
                #     'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + 'RIGHT_SHOULDER' + '].y * height)')
                # exec(
                #     'global x4;x4 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + 'LEFT_HIP' + '].x * width)')
                # exec(
                #     'global y4;y4 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + 'LEFT_HIP' + '].y * height)')
                # exec(
                #     'global x3;x3 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + 'RIGHT_HIP' + '].x * width)')
                # exec(
                #     'global y3;y3 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + 'RIGHT_HIP' + '].y * height)')
                # ppt = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                # cv2.fillPoly(frame, pts=[ppt], color=(220, 220, 220))

                for segment in segments:
                    exec(
                            'global x;x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment + '].x * width')
                    exec(
                            'global y;y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment + '].y * height')
                    radius = 3
                    center = (int(x), int(y))
                    color = (255, 200, 150)
                    thickness = 4
                    image = cv2.circle(frame, center, radius, color, thickness)

                for i in lines_left:
                    segment1 = i[0]
                    segment2 = i[1]
                    exec(
                        'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * width)')
                    exec(
                        'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * height)')
                    exec(
                        'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * width)')
                    exec(
                        'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * height)')
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 155), thickness=6)

                for i in lines_right:
                    segment1 = i[0]
                    segment2 = i[1]
                    exec(
                        'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * width)')
                    exec(
                        'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * height)')
                    exec(
                        'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * width)')
                    exec(
                        'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * height)')
                    cv2.line(frame, (x1, y1), (x2, y2), (155, 255, 155), thickness=6)

                for i in lines_center:
                    segment1 = i[0]
                    segment2 = i[1]
                    exec(
                        'global x1;x1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].x * width)')
                    exec(
                        'global y1;y1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment1 + '].y * height)')
                    exec(
                        'global x2;x2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].x * width)')
                    exec(
                        'global y2;y2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.' + segment2 + '].y * height)')
                    cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), thickness=6)
                
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=img)
                label_image.configure(image=photo)
                label_image.photo = photo

                if run == 0:
                    break

            except Exception as e:
                continue

    camera.release()

def workout(name, live = 1):
    if name == 'Wall Sit':
        if live == 1:
            static_workout_live((30, 40),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_HEEL', 'RIGHT_FOOT_INDEX'),
    ((20, 35),(12.5, 20),(15, 20),(20, 35)),
    ((25, 10),(40, 20),(25, 5),(15, 5)),
    ('Keep your back straight', 'Keep your thigh parallel to the ground', 'Keep your lower calf straight', 'keep your feet straight'),
    (('RIGHT_SHOULDER', 'RIGHT_HIP', 'y'), ('RIGHT_HIP', 'RIGHT_KNEE', 'x'), ('RIGHT_KNEE', 'RIGHT_ANKLE', 'y'),('RIGHT_HEEL', 'RIGHT_FOOT_INDEX', 'x')))

        elif live == 0:
            Thread(target=static_workout_upload, daemon=True, args = ((30, 40),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_HEEL', 'RIGHT_FOOT_INDEX'),
    ((20, 35),(12.5, 20),(15, 20),(20, 35)),
    ((25, 10),(40, 20),(25, 5),(15, 5)),
    ('Keep your back straight', 'Keep your thigh parallel to the ground', 'Keep your lower calf straight', 'keep your feet straight'),
    (('RIGHT_SHOULDER', 'RIGHT_HIP', 'y'), ('RIGHT_HIP', 'RIGHT_KNEE', 'x'), ('RIGHT_KNEE', 'RIGHT_ANKLE', 'y'),('RIGHT_HEEL', 'RIGHT_FOOT_INDEX', 'x')))).start()

    elif name == 'See Live Segmentation':
        if live == 1:
            embed_video_style_live()

    elif name == 'Pushup':
        if live == 1:
            dynamic_workout_live(40,('LEFT_WRIST', 'LEFT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER', 'LEFT_ANKLE'), (
    ('LEFT_WRIST', 'LEFT_SHOULDER', 'x'), ('RIGHT_WRIST', 'RIGHT_SHOULDER', 'x')), ((50, 70), (50, 70)), ((70, 80), (70, 80), (10, 15)), ((20, 20), (20, 20), (20, 20)), ('Your hands are too wide', 'Your hands are too wide', 'Keep your body symmetrical'), (('LEFT_ELBOW', 'LEFT_SHOULDER', 'y'), ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'y'), ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x')),
     ((10, 15),), ((20, 20),), ('Keep left and right arms symmetrical',), (('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x'),), ((10, 25), (20, 30), (10, 25), (20, 30), (10, 15)), ((20, 20),(20, 20),(20, 20),(20, 20),(20, 20)), ('Keep the body symmetrical', ), (('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x'), ),
        (('LEFT_WRIST', 'LEFT_SHOULDER'), ('RIGHT_WRIST', 'RIGHT_SHOULDER')), (('LEFT_WRIST', 'LEFT_SHOULDER'), ('RIGHT_WRIST', 'RIGHT_SHOULDER')), ((70, 80), (70, 80), (10, 15)), ((20, 20), (20, 20), (20, 20)), ('Your hands are too wide', 'Your hands are too wide', 'Keep your body symmetrical'), (('LEFT_ELBOW', 'LEFT_SHOULDER', 'y'), ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'y'), ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x')), ((25, 35), (20, 30), (25, 35), (20, 30), (10, 15)), ((20, 20), (20, 20), (20, 20), (20, 20), (20, 20)), ('Place your left hand towards inside', 'Place your left hand towards inside 2',
                 'Place your right hand towards inside', 'Place your right hand towards inside 2',
                 'Keep the body symmetrical'), (('LEFT_WRIST', 'LEFT_ELBOW', 'y'), ('RIGHT_WRIST', 'RIGHT_ELBOW', 'y'), ('LEFT_ELBOW', 'LEFT_SHOULDER', 'y'), ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'y'), ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x')), ('LEFT_WRIST','LEFT_SHOULDER', 'LEFT_SHOULDER', 'LEFT_ANKLE'))

        elif live == 0:
            Thread(target=dynamic_workout_upload, daemon=True, args = (40,('LEFT_WRIST', 'LEFT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER', 'LEFT_ANKLE'), (
    ('LEFT_WRIST', 'LEFT_SHOULDER', 'x'), ('RIGHT_WRIST', 'RIGHT_SHOULDER', 'x')), ((50, 70), (50, 70)), ((70, 80), (70, 80), (10, 15)), ((20, 20), (20, 20), (20, 20)), ('Your hands are too wide', 'Your hands are too wide', 'Keep your body symmetrical'), (('LEFT_ELBOW', 'LEFT_SHOULDER', 'y'), ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'y'), ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x')),
     ((10, 15),), ((20, 20),), ('Keep left and right arms symmetrical',), (('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x'),), ((10, 25), (20, 30), (10, 25), (20, 30), (10, 15)), ((20, 20),(20, 20),(20, 20),(20, 20),(20, 20)), ('Keep the body symmetrical', ), (('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x'), ),
        (('LEFT_WRIST', 'LEFT_SHOULDER'), ('RIGHT_WRIST', 'RIGHT_SHOULDER')), (('LEFT_WRIST', 'LEFT_SHOULDER'), ('RIGHT_WRIST', 'RIGHT_SHOULDER')), ((70, 80), (70, 80), (10, 15)), ((20, 20), (20, 20), (20, 20)), ('Your hands are too wide', 'Your hands are too wide', 'Keep your body symmetrical'), (('LEFT_ELBOW', 'LEFT_SHOULDER', 'y'), ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'y'), ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x')), ((25, 35), (20, 30), (25, 35), (20, 30), (10, 15)), ((20, 20), (20, 20), (20, 20), (20, 20), (20, 20)), ('Place your left hand towards inside', 'Place your left hand towards inside 2',
                 'Place your right hand towards inside', 'Place your right hand towards inside 2',
                 'Keep the body symmetrical'), (('LEFT_WRIST', 'LEFT_ELBOW', 'y'), ('RIGHT_WRIST', 'RIGHT_ELBOW', 'y'), ('LEFT_ELBOW', 'LEFT_SHOULDER', 'y'), ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'y'), ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'x')), ('LEFT_WRIST','LEFT_SHOULDER', 'LEFT_SHOULDER', 'LEFT_ANKLE'))).start()

    else:
        pass

    return()

def close_window():
    root.destroy()

####speak('I will be giving voice suggestions, press enter to close the program')

windll.shcore.SetProcessDpiAwareness(True)

#################  GUI  #################

window = 0

root=Tk()
root.title('Virtual Gym Trainer')
#root.attributes('-alpha', 0.9)
root.configure(bg = 'white')
root.iconbitmap('C:\\omega\\Playsign.ico')
root.attributes("-fullscreen", True)

##################  objects  ##################
topcolour = "#202020"

topframe = Frame(root, bd=0, bg = topcolour)
topframe.pack(fill=X)

texthead = 'Virtual Gym Trainer'
texthead = ttk.Label(topframe, text=texthead, foreground='#Ffd700', font=('Times', 20), background=topcolour)
close_button = Button(topframe, bg = topcolour, bd = 0, text="X", command=close_window, font=("fantasy", 14), foreground="#dddddd",pady = 6, padx = 15, anchor="center", activebackground="red")
close_button.pack(side=RIGHT, pady = 0, padx = 0)

#spacelabel = Label(topframe, bg=topcolour)
#spacelabel.pack(side=LEFT, padx = 104)
texthead.pack(pady=10, padx=10)

midcolour = "white"
colour = 'white'
midframedown = Frame(root, bd=0, bg = topcolour)

midframemain = Frame(root, bd=0, bg = 'white')
midframemain.pack(pady = 10, fill=BOTH, expand=1)
midframedown.pack(fill=X)

Frame_info = Frame(midframemain, bd=0, bg = 'white')
Scroll = Scrollbar(Frame_info)
Frame_info.pack(fill=Y, expand = 1)

canvas_info = Canvas(Frame_info, bg=colour, width = 1900, highlightthickness=0, yscrollcommand=Scroll.set)

frame2=Frame(canvas_info, bg = colour)
row1 = Frame(frame2, bg = 'white')
row1.pack(side = LEFT, fill=BOTH, padx = 12, expand=1)
row0 = Frame(frame2, bg = 'white')
row0.pack(side = LEFT, fill=BOTH, padx = 12, expand=1)

canvas_info.configure(yscrollcommand=Scroll.set, scrollregion="0 0 0 %s" % frame2.winfo_height())
canvas_info.pack(side=LEFT, fill=Y)

Scroll.config(command=canvas_info.yview)
Scroll.pack(side=RIGHT, fill=Y)

def end_workout(upload = 0):
    global run, uploaded
    run = 0
    midframe1.pack_forget()
    midframe2.pack(pady = 10, fill=BOTH, expand=1)
    if upload == 0:
        uploaded = "C:\\Users\\vaibh\\Desktop\\Exercise form\\out.mp4"

    else:
        uploaded = upload

def home():
    midframe2.pack_forget()
    Frame_info.pack(fill=Y, expand=1)

def gen_report():
    global uploaded, camera, current_workout
    camera = cv2.VideoCapture(uploaded)
    workout(current_workout, 0)

def save_pdf():
    download_file = filedialog.asksaveasfilename(filetypes=(("Video", "*.mp4"),))
    if download_file == '':
        download_file = 'C:\\Users\\vaibh\\Desktop\\Exercise form\\Advance report.mp4'
    else:
        download_file += '.pdf'

######## EXTRA WINDOWS ##########

## window 1

midframe1 = Frame(midframemain, bd=0, bg = 'white')
label_image = Label(midframe1, width = 1280, height = 720)
label_image.pack(side = LEFT, pady = 20, padx = 10)
frame_info2 = Frame(midframe1, bd=0, bg = "white")
frame_info2.pack(side=LEFT, expand = 1, padx = 10, pady = 30, fill=BOTH)

bg_colour = 'white'

time = 'Workout time : 0.0'
message = 'Message : '
calories = "Calories burnt : 0.0"
reps = "Total Reps : 0"
name = "Workout"
heading_label = Label(frame_info2, bg = "#202020", bd = 0, text=name, font=("fantasy", 20, 'bold'), foreground="#dddddd",pady = 10, padx = 15, anchor="center")
heading_label.pack(pady = 15, padx = 10, fill=X)
time_label = Label(frame_info2, bg = bg_colour, bd = 0, text=time, font=("fantasy", 14), foreground="#202020",pady = 6, padx = 15, anchor="center")
time_label.pack(pady = 5, padx = 10, fill=X)
reps_label = Label(frame_info2, bg = bg_colour, bd = 0, text=reps, font=("fantasy", 14), foreground="#202020",pady = 6, padx = 15, anchor="center")
reps_label.pack(pady = 5, padx = 10, fill=X)
calories_label = Label(frame_info2, bg = bg_colour, bd = 0, text=calories, font=("fantasy", 14), foreground="#202020",pady = 6, padx = 15, anchor="center")
calories_label.pack(pady = 5, padx = 10, fill=X)
message_label = Label(frame_info2, bg = bg_colour, bd = 0, text=message, font=("fantasy", 14), foreground="#202020",pady = 6, padx = 15, anchor="center")
message_label.pack(pady = 5, padx = 10, fill=X)
position_label = Label(frame_info2, bg = bg_colour, bd = 0, text='Position : ', font=("fantasy", 14), foreground="#202020",pady = 6, padx = 15, anchor="center")
position_label.pack(pady = 5, padx = 10, fill=X)
end_button = Button(frame_info2, bg = "#252525", bd = 0, text="End Workout", command=end_workout, font=("fantasy", 14), foreground="#d0d0d0",pady = 6, padx = 15, anchor="center", activebackground="green")
end_button.pack(side=BOTTOM, pady = 0, padx = 0)

## window 2
midframe2 = Frame(midframemain, bd=0, bg = bg_colour)
home_page = Button(midframe2, bg = "#019020", bd = 0, text="Continue working out", command=home, font=("fantasy", 14), foreground="#d0d0d0",pady = 6, padx = 15, anchor="center", activebackground="#888888")
home_page.pack(pady = 10, padx = 0)
generate_adv_report = Button(midframe2, bg = "#121212", bd = 0, text="More accurate video report", command=gen_report, font=("fantasy", 14), foreground="#d0d0d0",pady = 6, padx = 15, anchor="center", activebackground="#888888")
generate_adv_report.pack(pady = 10, padx = 0)
download_pdf = Button(midframe2, bg = "#121212", bd = 0, text="Download PDF", command=save_pdf, font=("fantasy", 14), foreground="#d0d0d0",pady = 6, padx = 15, anchor="center", activebackground="#888888")
status_label = Label(midframe2, bg = "#cccccc", bd = 0, text='', font=("fantasy", 14), foreground="#d0d0d0",pady = 6, padx = 15, anchor="center")
status_label.pack(pady = 10, fill=X)
hp_bar2 = ttk.Progressbar(midframe2, orient=HORIZONTAL,mode='determinate')
hp_bar2.pack(pady = 20, fill=X)

## adding info
def specific_workout(name):
    global current_workout
    global window
    global run
    current_workout = name[0]
    run = 1
    window = 1
    if name[1] == 'upload':
        upload(current_workout)

    elif name[1] == 'live':
        live_workout(current_workout)

def live_workout(name):
    global camera
    camera = cv2.VideoCapture(0)
    Frame_info.pack_forget()
    window = 1
    heading_label.config(text = name)
    midframe1.pack(pady = 10, fill=BOTH, expand=1)
    show_frame = Thread(target = workout, daemon = True, args = (name,))
    show_frame.start()

def upload(name):
    Frame_info.pack_forget()
    path = filedialog.askopenfilename()
    end_workout(path)

def add(num,item, desc, parts, img):
    introw = num%2
    back = '#eeeeee'
    exec("globals()[\"frame_info\"+str(num)] = Frame(row"+ str(introw) +", bg = back)")
    exec("frame_info" + str(num) + ".pack(pady = 12, fill=X, expand=1)")
    try:
        img = Image.open(img)
    except:
        img = Image.open('files\\playsign.png')
    width, height = img.size
    img = img.resize((350, 350), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
        
    exec("globals()[\"image_\"+str(num)] = Label(frame_info" + str(
        num) + ",bg = back, bd = 0, image=img, padx = 12, anchor=\'w\')")
    exec("image_" + str(num) + ".image = img")
    exec("image_" + str(num) + ".pack(side = LEFT, pady = 12, padx = 12)")
    exec("globals()[\"frame_right\"+str(num)] = Frame(frame_info" + str(num) + ", bg = back)")
    exec("frame_right" + str(num) + ".pack(side=RIGHT, pady = 0, fill=BOTH, expand=1)")
    exec("globals()[\"Name\"+str(num)] = Label(frame_right" + str(
        num) + ",bg = back, bd = 0, text=item, font=(\'fantasy\', 14, 'bold'), foreground=\'black\',pady = 10, padx = 15, anchor=\'center\')")
    exec("Name" + str(num) + ".pack(fill=X)")
    exec("globals()[\"desc\"+str(num)] = Label(frame_right" + str(
        num) + ",bg = back, bd = 0, text=desc, font=(\'fantasy\', 12, 'bold'), foreground=\'#bb0000\', wraplength = 468, padx = 10, anchor=\'center\')")
    exec("desc" + str(num) + ".pack(fill=X, pady = 12)")
    exec("globals()[\"parts\"+str(num)] = Label(frame_right" + str(
        num) + ",bg = back, bd = 0, text=parts, font=(\'fantasy\', 12, 'bold'), foreground=\'#019020\', wraplength = 468, padx = 10, anchor=\'center\')")
    exec("parts" + str(num) + ".pack(fill=X, pady = 12)")
    exec("globals()[\"frame_down\"+str(num)] = Frame(frame_right" + str(num) + ", bg = back, pady = 10)")
    exec("frame_down" + str(num) + ".pack(side=BOTTOM, pady = 0, expand=0)")
    exec("globals()[\"start\"+str(num)] = Button(frame_down" + str(
        num) + ",bg = \'#019020\', bd = 0, text=\'Live workout\', command=partial(specific_workout, (item, \'live\')),font=(\'fantasy\', 13), foreground=\'#ffffff\',pady = 6, padx = 15, anchor=\'center\', activebackground=\'green\')")
    exec("start" + str(num) + ".pack(side=RIGHT, pady = 0, padx = 10)")
    exec("globals()[\"upd\"+str(num)] = Button(frame_down" + str(
        num) + ",bg = \'#121212\', bd = 0, text=\'Upload video\', command=partial(specific_workout, (item, \'upload\')),font=(\'fantasy\', 13), foreground=\'#d0d0d0\',pady = 6, padx = 15, anchor=\'center\', activebackground=\'green\')")
    exec("upd" + str(num) + ".pack(side=RIGHT, pady = 0, padx = 0)")

## static workouts to add : plank, wall squats, side planks
## two position workouts : Pushups, sit-ups, bicep curls, bench press
workouts = [('Plank', 'The plank (also called a front hold, hover, or abdominal bridge) is an isometric core strength exercise that involves maintaining a position similar to a push-up for the maximum possible time', 'Core, upper body, Little of lower body', 'files\\plank.png'),
            ('Wall Sit', 'A wall squat, also known as a wall sit, is a bodyweight exercise that targets muscles in your core and lower body.', ' Quads, Glutes - Primarily and Calves, Abdominals, Lower Back, Hamstrings secondary','files\\wall squat.jpg'),
            ('Side_plank', 'The side plank is helps to work the two layers of muscle along the sides of your core, (obliques). These muscles help you rotate and bend your trunk', 'Shoulders, hips, and sides of your core', 'files\\side plank.jpg'),
            ('Pushup', 'The push-up builds both upper-body and core strength. It has many modifications', 'deltoids, the pectoral muscles, the triceps and biceps, hip muscles, erector spinae', 'files\\pushups.png'),
            ('Sit_ups', 'The sit-up is an abdominal endurance training exercise to strengthen, tighten and tone the abdominal muscles. It is similar to a crunch, but sit-ups have a fuller range of motion', 'rectus abdominis, transverse abdominis, obliques, hip flexors, chest, and neck', 'files\\sit up.jpeg'),
            ('Bicep_Curl', 'Bicep curl targets the biceps brachii muscle. It may be performed using a barbell, dumbbell, resistance band, or any other equipments as well', 'Biceps Brachii', 'files\\bicep curl.png'),
            ('Bench_Press', 'The bench press or chest press is a weight training exercise to work your upper body', 'pectoralis major, anterior deltoid, triceps brachii, biceps brachii, serratus anterior', 'files\\benchpress.png'),
            ('See Live Segmentation', 'Live skeletal view', '-', 'None')]

for n, i in enumerate(workouts):
    add(n+1, i[0], i[1], i[2], i[3])

canvas_info.create_window((0, 0), window=frame2, anchor='nw', width = 1900)
root.update()
canvas_info.configure(scrollregion="1 1 1 %s" % frame2.winfo_height())
frame_user = Frame(midframedown, bd=0, bg = midcolour)
#frame_user.pack(padx = 10, fill=X)
line1 = Frame(frame_user, bd=2, bg = midcolour)
line2 = Frame(frame_user, bd=2, bg = midcolour)
line1.pack(fill=X)
line2.pack(fill=X)

username = Label(line1, bg = midcolour, text='Vaibhav', font=('fantasy', 15), foreground = 'green')
height = Label(line1, bg = midcolour, text='180 cm', font=('fantasy', 15), foreground = 'green')
weight = Label(line1, bg = midcolour, text='75 Kg', font=('fantasy', 15), foreground = 'green')
calories_burnt = Label(line2, bg = midcolour, text='This week : 400 Kcal', font=('fantasy', 14), foreground = 'green')
time_worked = Label(line2, bg = midcolour, text='Today : 100 mins', font=('fantasy', 14), foreground = 'green')

username.pack(side=LEFT, expand=True)
height.pack(side=LEFT, expand=True)
weight.pack(side=LEFT, expand=True)
calories_burnt.pack(side=LEFT, expand=True)
time_worked.pack(side=LEFT, expand=True)

##### MUSIC PART

def start():
    global n
    global music
    playsong(n)

def playsong(numb):
    global music
    global a
    mixer.music.unload()
    music = songs[numb]
    mixer.music.load(music)
    clip = audio(music)
    a = clip.duration
    clip.close()
    a = round(a)
    mixer.music.play()

def addsongs():
    global songs
    global n

    songs = listdir()
    songsfiltered = list()
    for song in songs:
        if song.split('.')[-1]=='mp3':
            songsfiltered.append(song)

        else:
            pass

    songs = songsfiltered

    n = 0
    
    shuffle(songs)

    start()

def chpath():
        global path
        path = filedialog.askdirectory()
        chdir(path)
        addsongs()

def playnext():
    global n
    if n == len(songs)-1:
        n = 0

    else:
        n = n+1
    playsong(n)

def playprev():
    global n
    if n == 0:
        n = len(songs)-1
    else:
        n = n-1
    playsong(n)

def playpause():
    global music
    specific(music.split(".mp3")[0])

def running():
    global music
    global a
    global n
    global vol
    new = n+1
    mixer.music.queue(songs[new])
    status(music, a)
    old = music

    while True:
        sleep(0.9)

        for events in event.get():
            if events.type == MUSIC_END:
                if n == len(songs)-1:
                    n = 0

                else:
                    n = n+1

                music = songs[n]

        if music == 'kill':
            mixer.music.stop()
            mixer.music.unload()
            quit()

        if music == 'kill2':
            break

        if music != old:
            new = n+1
            clip = audio(music)
            a = clip.duration
            clip.close()
            a = round(a)
            status(music,a)
            if new == len(songs):
                new = 0

            mixer.music.queue(songs[new])
            old = music

        vol = w.get()
        tpl = mixer.music.get_pos()
        updlen(round(tpl/a/10),round(tpl),vol/100)

def status(music,length):
    length = str(length//60) + ':' + str(length%60)
    tmusic.config(text=music)
    tlen.config(text=length)

def updlen(l,elapsed, vol):
    t2 = Thread(target=updlength, args=(l,elapsed,vol))
    t2.start()

def updlength(l,elapsed,vol):
    try:
        global a
        hp_bar1['value'] = l
        elapsed = round(elapsed/1000)
        played = a - elapsed
        elapsed = str(elapsed//60) + ':' + str(elapsed%60)
        played = str(played//60) + ':' + str(played%60)
        mixer.music.set_volume(vol)
        tela.config(text=elapsed)
        tlen.config(text=played)
    except:
        pass

def specific(songname):
    songname = songname + '.mp3'
    global n
    global songs
    newn = songs.index(songname)
    if newn == n:
        if (mixer.music.get_busy()) == True:
            mixer.music.pause()
            playbutton2.config(image=play2)

        else:
            mixer.music.unpause()
            playbutton2.config(image=pause)

    if n != newn:
        playbutton2.config(image=pause)
        n = newn
        start()

t1 = Thread(target = running, daemon=True)
play2 = Image.open('C:\\omega workout\\Playsign2.png')
pause = Image.open("C:\\omega workout\\pause.png")
play3 = Image.open('C:\\omega workout\\Playsignre.png')
originalpath = 'C:\\omega workout\\online player files'
path = 'C:\\omega workout\\online player files'
chdir(path)
imgpath = originalpath + "\\images"

init()
MUSIC_END = USEREVENT+1
mixer.music.set_endevent(MUSIC_END)
mixer.music.set_volume(0.75)
n = 0
a = 0
music = ''

lowcolour = '#252525'

lowframe = Frame(midframedown, bd=0, bg = lowcolour)
lowframe.pack(fill=X)

infoframe = Frame(lowframe, bd=0, bg = lowcolour)

tmusic = music
tmusic = ttk.Label(infoframe, text=tmusic, foreground='gold', font=('fantasy', 12, 'bold'), background=lowcolour)

s = ttk.Style()
s.theme_use("default")
s.configure("TProgressbar", thickness=8, background='orange', troughcolor='white', bd = 0, highlightthickness=0, borderwidth=0)

hp_bar1 = ttk.Progressbar(lowframe, orient=HORIZONTAL,mode='determinate', style = 'TProgressbar')

tela = '00:00'
tela = ttk.Label(infoframe, text=tela, foreground='gold', font=('fantasy', 10, 'bold'), background=lowcolour)

tlen = str(a//60) + ':' + str(a%60)
tlen = ttk.Label(infoframe, text=tlen, foreground='gold', font=('fantasy', 10, 'bold'), background=lowcolour)

hp_bar1.pack(fill=X)

tela.pack(side=LEFT, padx = 20)
tlen.pack(side=RIGHT, padx = 20)
tmusic.pack(pady = 5)
infoframe.pack(fill=X)

fcolour = '#252525' #'#44cc88'

functionframe = Frame(midframedown, bd=0, bg = fcolour, pady = 5)
functionframe.pack(fill=X)

f2colour = 'black'

functionframe2 = Frame(functionframe, bg = f2colour, padx = 15, pady = 2)

shuff = Button(functionframe, text='Shuffle', bg='black', bd=0, command=addsongs, font=('fantasy', 12), foreground='#eeeeee', pady = 15, padx = 35)
songsbutton = Button(functionframe, text='Songs', bg='black', bd=0, command=chpath, font=('fantasy', 12), foreground='#eeeeee', pady = 15, padx = 35)

prevbutton = Button(functionframe2, text='<', bg=f2colour, bd=0, command=playprev, font=('fantasy', 17), foreground='#dddddd', padx = 13)
nextbutton = Button(functionframe2, text='>', bg=f2colour, bd=0, command=playnext, font=('fantasy', 17), foreground='#dddddd', padx = 13)
#Button(root, text='<<', bg=colour2, bd=0, command=back, font=('fantasy', 15), foreground='#dddddd').place(x=795, y=910, width=50, anchor = CENTER)
#Button(root, text='>>', bg=colour2, bd=0, command=forw, font=('fantasy', 15), foreground='#dddddd').place(x=1095, y=910, width=50, anchor = CENTER)

pause = pause.resize((55,65), Image.ANTIALIAS)
pause = ImageTk.PhotoImage(pause)

play2 = play2.resize((55,65), Image.ANTIALIAS)
play2 = ImageTk.PhotoImage(play2)
playbutton2 = Button(functionframe2, image=pause, bg=f2colour, command=playpause, bd = 0, activebackground='#cccccc')

play3 = play3.resize((90,70), Image.ANTIALIAS)
play3 = ImageTk.PhotoImage(play3)

w = Scale(functionframe, from_=0, to=100, orient = HORIZONTAL, bg=lowcolour, fg='#dddddd', bd = 0, troughcolor='blue', sliderlength=20, highlightbackground = lowcolour, length = 220)
w.set(75)

Label(functionframe, text='Vol : ', foreground='#dddddd', font=('fantasy', 10, 'bold'), background=lowcolour).pack(side=LEFT, padx = 16, anchor='e')
w.pack(side=LEFT, anchor='w')
shuff.pack(side=RIGHT, padx = 20)
songsbutton.pack(side=RIGHT)

functionframe2.pack()
prevbutton.pack(side=LEFT)
nextbutton.pack(side=RIGHT)
playbutton2.pack(padx = 25)

addsongs()
playpause()
t1.start()

def spacepress(arg):
    playpause()

root.bind('<space>', spacepress)

root.mainloop()
run = 0
music = 'kill'
