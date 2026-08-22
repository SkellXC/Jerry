import cv2
import numpy as np
import mss
import pygetwindow as gw
from ultralytics import YOLO

# 1. Load the YOLO11 nano model
model = YOLO('yolo11n.pt')

# 2. Locate the Scrcpy Window
WINDOW_TITLE = "ScrcpyCam"
windows = gw.getWindowsWithTitle(WINDOW_TITLE)

if not windows:
    print(f"Error: Could not find a window named '{WINDOW_TITLE}'.")
    print("Make sure you launched Scrcpy using: scrcpy --window-title \"ScrcpyCam\"")
    exit()

scrcpy_window = windows[0]

# State tracking variable
PERSON_PRESENT = False

print("Starting room surveillance... Press 'q' on the video window to quit.")

# 3. Setup MSS for rapid screen capture
with mss.mss() as sct:
    while True:
        # Dynamically grab the window coordinates every frame 
        # (This allows you to move the Scrcpy window around your screen without breaking the feed)
        monitor = {
            "top": scrcpy_window.top,
            "left": scrcpy_window.left,
            "width": scrcpy_window.width,
            "height": scrcpy_window.height
        }

        # Capture the raw pixels from that specific region
        sct_img = sct.grab(monitor)

        # Convert the raw image into a format OpenCV and YOLO can read (BGRA to BGR)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 4. Run object detection on the frame [cite: 356]
        results = model(frame, classes=0, conf=0.5, verbose=False)

        person_detected_in_frame = False

        # 5. Process the results [cite: 356]
        for r in results:
            if len(r.boxes) > 0:  
                person_detected_in_frame = True
                
                # Plot the bounding boxes onto the frame [cite: 356]
                frame = r.plot()

        # 6. Logic to handle terminal output and state [cite: 357]
        if person_detected_in_frame:
            if not PERSON_PRESENT:
                print("Status: Person entered the room.")
                PERSON_PRESENT = True
        else:
            if PERSON_PRESENT:
                print("Status: Room is empty.")
                PERSON_PRESENT = False

        # Display the video feed
        cv2.imshow('Room Surveillance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()