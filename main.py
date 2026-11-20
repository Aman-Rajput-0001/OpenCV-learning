import cv2
import numpy
import mediapipe as mp
import time
import pyautogui
import math

pyautogui.PAUSE = 0
# # print(cv2.__version__)
# camera = cv2.VideoCapture(0)
# ret , frame = camera.read()
# cv2.imshow("My App",frame)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
modal_path = r"D:\Aman-Learning\OpenCV learning\hand_landmarker.task"
Options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=modal_path),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(Options)
camer = cv2.VideoCapture(0)
# Set camera resolution
camer.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camer.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not camer.isOpened():
    print("camera could not opened")
    landmarker.close()
    exit()
printed = False
l = False
m = False
x = 100
x_values = []
y_values = []
point = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]
screen_width, screen_hight = pyautogui.size()
prev_x = None
prev_y = None
previous_pinching = False
alpha = 0.45
frameR = 100  # Active camera area margin
# y = 100
# print(mp.__version__)
# print(mp.solutions.hands)
while True:
    # y = y+5
    # check if camera is opened  --->True/False
    ret, frame = camer.read()
    # check Frame is captured sucessfull ----> True/False
    if not ret:
        break
    # Mirror camera frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestmap_ms = time.monotonic_ns() // 1_000_000
    result = landmarker.detect_for_video(mp_image, timestmap_ms)
    # Get frame dimensions
    height, width = frame.shape[:2]
    # Draw active camera region
    cv2.rectangle(
        frame, (frameR, frameR), (width - frameR, height - frameR), (255, 255, 0), 2
    )
    # To detect full hand on camera
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        points = []
        # Convert normalized coordinates into pixels
        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            # print(x,y)
            points.append((x, y))
            # Draw landmark point
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # Connect landmarks
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
        # Get thumb and index fingertips
        thumb_tip = points[4]
        index_tip = points[8]
        # Calculate distance between two fingertips
        pinch_distance = math.dist(thumb_tip, index_tip)
        print(pinch_distance)
        # Calculate palm width
        palm_width = math.dist(points[5], points[17])
        # Detect pinch gesture
        is_pinching = pinch_distance < (palm_width * 0.35)
        # Trigger click only once per pinch
        if is_pinching and not previous_pinching:
            pyautogui.click()
            # print("LEFT CLICK")
        previous_pinching = is_pinching
        # Visual feedback
        cv2.line(frame, thumb_tip, index_tip, (0, 255, 255), 3)
        cv2.circle(frame, thumb_tip, 8, (255, 0, 255), -1)
        cv2.circle(frame, index_tip, 8, (255, 0, 255), -1)
        # Get index fingertip
        index_finger = landmarks[8]
        # Convert camera coordinates into screen coordinates using active region
        camera_x = int(index_finger.x * width)
        camera_y = int(index_finger.y * height)
        target_x = int((camera_x - frameR) * screen_width / (width - 2 * frameR))
        target_y = int((camera_y - frameR) * screen_hight / (height - 2 * frameR))
        # Keep coordinates inside screen boundaries
        target_x = max(0, min(target_x, screen_width - 1))
        target_y = max(0, min(target_y, screen_hight - 1))
        # Apply EMA smoothing
        if prev_x is None:
            smooth_x = target_x
            smooth_y = target_y
        else:
            smooth_x = int(alpha * target_x + (1 - alpha) * prev_x)
            smooth_y = int(alpha * target_y + (1 - alpha) * prev_y)
        # Move mouse cursor
        pyautogui.moveTo(smooth_x, smooth_y)
        prev_x = smooth_x
        prev_y = smooth_y
        # Draw index fingertip separately
        cv2.circle(frame, points[8], 10, (0, 0, 255), -2)
    else:
        # Reset smoothing when hand is not detected
        prev_x = None
        prev_y = None
        previous_pinching = False
    cv2.putText(
            img=frame,  # Jis frame par likhna hai
            text="Air Mouse",  # Jo text dikhana hai
            org=(50, 50),  # Text ki position (X, Y coordinates)
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font ka style
            fontScale=1,  # Text ka size
            color=(0, 255, 0),  # Text ka color (BGR format: yahan Green hai)
            thickness=2,  # Text ki motai (boldness)
        )
    cv2.imshow("Air MOuse", frame)
    if not m:
        height, width, channel = frame.shape
        x = int(width / 4)
        y = int(height / 2)
        # print(centre_x, centre_y)
        m = True
    if not l:
        pix = frame[100, 200]
        print(f" B - {pix[0]}", end=" ")
        print(f" G - {pix[1]}", end=" ")
        print(f" R - {pix[2]}")
        l = True
    # Details Of frame each
    if not printed:
        # print(type(frame)) ----- Datatype is numpyArray
        print(f" The height,width and colour {frame.shape}")
        print("=" * 100)
        print(f"frame is captured status: {ret}")
        print("Screen:", screen_width, screen_hight)
        print("Camera:", width, height)
        printed = True
    # press "q" to close camera or all window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camer.release()
landmarker.close()
cv2.destroyAllWindows()
