
import cv2
import numpy as np
import math
from datetime import datetime
import PySimpleGUI as sg
import tensorflow as tf
import face_recognition
import os

path = 'ImagesBasic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

detect=True
def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Put in try block if list index is out of range
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            global detect
            detect=False
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while detect:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
cap.release()
def get_landmark_model(saved_model='models/pose_model'):

    model = tf.saved_model.load("C:\\Users\\ashwi\\OneDrive\\Desktop\\Proctoring-AI\\models\pose_model")
    return model

def find_faces(img, model):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

def get_face_detector(modelFile=None,
                      configFile=None,
                      quantized=False):
    if quantized:
        if modelFile == None:
            modelFile = "models/opencv_face_detector_uint8.pb"
        if configFile == None:
            configFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    else:
        if modelFile == None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile == None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def move_box(box, offset):
    """Move the box to direction specified by vector offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]


def detect_marks(img, model, face):
    offset_y = int(abs((face[3] - face[1]) * 0.1))
    box_moved = move_box(face, [0, offset_y])
    facebox = get_square_box(box_moved)

    h, w = img.shape[:2]
    if facebox[0] < 0:
        facebox[0] = 0
    if facebox[1] < 0:
        facebox[1] = 0
    if facebox[2] > w:
        facebox[2] = w
    if facebox[3] > h:
        facebox[3] = h

    face_img = img[facebox[1]: facebox[3],
               facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # # Actual detection.
    predictions = model.signatures["predict"](
        tf.constant([face_img], dtype=tf.uint8))

    # Convert predictions to landmarks.
    marks = np.array(predictions['output']).flatten()[:136]
    marks = np.reshape(marks, (-1, 2))

    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]
    marks = marks.astype(np.uint)

    return marks

def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]

def eye_on_mask(mask, side, shape):

    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1 ] +points[2][1] )//2
    r = points[3][0]
    b = (points[4][1 ] +points[5][1] )//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):

    x_ratio = (end_points[0] - cx ) /(cx - end_points[2])
    y_ratio = (cy - end_points[1] ) /(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0


def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8]) // 2
    x = point_2d[2]

    return (x, y)
def contouring(thresh, mid, img, end_points, right=False):

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10' ] /M['m00'])
        cy = int(M['m01' ] /M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass

def process_thresh(thresh):

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh

num_of_times=0
def print_eye_pos(img, left, right):
    global num_of_times
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking left')
            text = 'Looking left'
            num_of_times += 1
        elif left == 2:
            print('Looking right')
            text = 'Looking right'
            num_of_times += 1
        elif left == 3:
            print('Looking up')
            text = 'Looking up'

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (30, 30), font,
                    1, (0, 255, 255), 2, cv2.LINE_AA)





face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX
thresh = img.copy()
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)
# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

last_detected = datetime.now()
while(True):
    ret, img = cap.read()
    rects = find_faces(img, face_model)

    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)
        marks = detect_marks(img, landmark_model, rect)
        # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
        image_points = np.array([
            marks[30],  # Nose tip
            marks[8],  # Chin
            marks[36],  # Left eye left corner
            marks[45],  # Right eye right corne
            marks[48],  # Left Mouth corner
            marks[54]  # Right mouth corner
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)
        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)


        try:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90

        try:
            m = (x2[1] - x1[1]) / (x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1 / m)))
        except:
            ang2 = 90

            # print('div by zero error')
        if ang1 >= 48:
            print('Head down')
            cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
        elif ang1 <= -48:
            print('Head up')
            cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

        if ang2 >= 48:
            print('Head right')
            cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
        elif ang2 <= -48:
            print('Head left')
            cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)


        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
        print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
        if num_of_times > 10:
            last_detected = datetime.now()
            print("PLease stop looking away")
            num_of_times = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            txt="Please abstain from looking elsewhere"
            if (datetime.now() - last_detected).total_seconds() < 5:
                cv2.putText(img, txt, (20, 55), font,
                    1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
