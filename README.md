# 🖱️ AI Virtual Air Mouse (Computer Vision & Gesture Control)

An intelligent, real-time **AI Virtual Mouse** built with Python, OpenCV, and Google MediaPipe Tasks API. Control your system's mouse cursor and perform clicks entirely touch-free through hand gestures captured via your webcam!

> 🚀 **Author's Note:**  
> This is my **first project build while learning OpenCV and Computer Vision**! I'm diving deep into CV, image processing, and human-computer interaction. Stay tuned—this is just the beginning, and I will be building many more crazy and exciting AI/CV projects soon! 🔥

---

## ✨ Features

- **🎯 Real-Time Hand Landmark Tracking:** Leverages the latest MediaPipe Vision Tasks API (`HandLandmarker`) with 21 3D hand keypoints for high precision and low latency.
- **✨ Butter-Smooth Cursor Motion (EMA Smoothing):** Implements Exponential Moving Average filtering ($\alpha = 0.45$) to eliminate hand tremors and jitter for steady pointing.
- **📐 Active Camera Region (Comfort Margin):** Configured with an active frame margin (`frameR = 100`) so you can reach every corner of your screen without stretching your hand out of camera view.
- **👌 Adaptive Pinch-to-Click:** Dynamic distance calculation between thumb and index fingertips, automatically scaled by palm width ($5 \to 17$ landmark distance) to ensure consistent clicking regardless of how close or far your hand is from the camera.
- **🛡️ Single-Trigger Click Logic:** State-machine debounce prevents accidental repeated clicks during a single pinch gesture.
- **🎨 Interactive Visual Feedback:** Live HUD displaying hand skeleton connections, active boundary box, fingertip indicators, and pinch connection line.
- **🪞 Mirrored Feed:** Camera display is flipped horizontally for natural, mirror-like control.

---

## 🧩 What is `hand_landmarker.task` and Why is it Needed?

The file **`hand_landmarker.task`** is the pre-trained neural network bundle that acts as the **"brain"** of this project.

### 🔍 Detailed Explanation:

1. **What is it?**  
   It is Google MediaPipe’s packaged binary model asset (built with TensorFlow Lite) specifically designed for the **MediaPipe Tasks Vision API** (`HandLandmarker`). It packages all required weights, graphs, and metadata into a single `.task` file.

2. **What does it do? (Two-Stage ML Pipeline):**
   - **Stage 1 (Hand Detector):** Scans the input video frame to detect the presence of human hands and crops out a Region of Interest (ROI).
   - **Stage 2 (Landmark Estimation):** Analyzes the cropped hand region to predict **21 high-precision 3D coordinates ($X, Y, Z$)** for every knuckle, joint, and fingertip in real-time.

3. **Why do we need it in this project?**
   - In `main.py`, the hand landmarker is initialized with:
     ```python
     BaseOptions = mp.tasks.BaseOptions
     HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
     Options = HandLandmarkerOptions(
         base_options=BaseOptions(model_asset_path=modal_path),
         running_mode=RunningMode.VIDEO,
         ...
     )
     ```
   - Without `hand_landmarker.task`, MediaPipe has no model weights to evaluate and cannot track hand keypoints, resulting in an initialization error.
   - It runs completely **locally and offline** at high frames per second (FPS) without requiring any cloud API or internet connection.

---

## 🛠️ Tech Stack & Libraries

- **Python 3.x**
- **[OpenCV](https://opencv.org/) (`opencv-python`):** Video capture, image transformations (BGR to RGB, flipping), HUD drawing (bounding box, landmarks, lines).
- **[MediaPipe](https://developers.google.com/mediapipe) (`mediapipe`):** Google's high-speed Hand Landmarker vision task for 21 hand points.
- **[PyAutoGUI](https://pyautogui.readthedocs.io/):** Native OS mouse movement and click event automation.
- **NumPy & Math:** Geometric coordinate calculation, Euclidean distance, and screen mapping.

---

## 📁 Project Structure

```text
OpenCV learning/
│
├── main.py                 # Core AI Mouse script
├── hand_landmarker.task    # Pre-trained MediaPipe Hand Landmarker model asset
├── .gitignore              # Ignored files (temporary & runner files)
├── README.md               # Project documentation & guide
```

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have Python 3.8+ installed on your system.

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install opencv-python mediapipe pyautogui numpy
```

### 3. MediaPipe Task Model

Ensure `hand_landmarker.task` is placed in your project directory.  
If not downloaded yet, you can get it from Google's official MediaPipe repository:
- [Download hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

*(Note: Verify that `modal_path` in `main.py` points to your `hand_landmarker.task` file location).*

### 4. Run the Project

Run the application:

```bash
python main.py
```

Press **`q`** while the camera window is focused to exit.

---

## 🖐️ Gestures & Controls

| Gesture | Landmark Points | Action |
| :--- | :--- | :--- |
| **Move Cursor** | Point with your **Index Finger** (Landmark 8) | Moves cursor smoothly across the desktop screen |
| **Left Click** | **Pinch** Thumb Tip (4) and Index Tip (8) together | Performs a single left mouse click |
| **Quit App** | Press **`q`** key on keyboard | Closes camera feed and cleans up resources |

---

## 🧠 How It Works Behind the Scenes

1. **Frame Capture & Mirroring:**  
   OpenCV captures frames at 1280x720 and flips them horizontally using `cv2.flip(frame, 1)` so your movements match what you see on screen.

2. **Hand Tracking:**  
   The frame is converted to RGB and passed to MediaPipe's `HandLandmarker.detect_for_video()`.

3. **Coordinate Mapping & Clamping:**  
   The index fingertip coordinate is translated from webcam space to screen resolution (`pyautogui.size()`) taking the active margin into account:
   $$\text{target\_x} = (\text{cam\_x} - \text{frameR}) \times \frac{\text{screen\_width}}{\text{cam\_width} - 2 \times \text{frameR}}$$

4. **Exponential Smoothing (EMA):**  
   To prevent shaky mouse movement:
   $$\text{smooth} = \alpha \cdot \text{target} + (1 - \alpha) \cdot \text{previous}$$

5. **Scale-Invariant Pinch Detection:**  
   Instead of a static pixel threshold, pinch sensitivity scales with hand distance:
   $$\text{pinch\_threshold} = \text{palm\_width} \times 0.35$$

---

## 🌟 Future Roadmap & Upcoming Projects

Since this is the foundation of my OpenCV journey, here are some upcoming builds and features in development:

- [ ] Right-click, double-click, and drag-and-drop gestures
- [ ] Smooth two-finger scrolling (up/down)
- [ ] Volume and brightness control through hand elevation / thumb-index gestures
- [ ] Virtual Drawing Canvas & Air Canvas
- [ ] Face mesh & eye-tracking cursor navigation
- [ ] Sign language & custom action recognizer

---

## 💡 Author

Built with curiosity and passion by **Aman Singh** while exploring the fascinating world of Computer Vision & OpenCV! 🚀
