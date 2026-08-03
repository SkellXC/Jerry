import cv2
import mediapipe as mp
import face_recognition

# 1. Load reference image
reference_image = face_recognition.load_image_file("my_face.jpg")
my_face_encoding = face_recognition.face_encodings(reference_image)[0]

# 2. Setup MediaPipe Detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cam = cv2.VideoCapture(0)

frameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, 20.0, (frameWidth, frameHeight))

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Detect faces with MediaPipe
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        h, w, _ = frame.shape
        face_locations = []
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Convert MediaPipe's normalized coordinates to dlib's pixel format (top, right, bottom, left)
            top = max(0, int(bbox.ymin * h))
            bottom = min(h, int((bbox.ymin + bbox.height) * h))
            left = max(0, int(bbox.xmin * w))
            right = min(w, int((bbox.xmin + bbox.width) * w))
            
            face_locations.append((top, right, bottom, left))
            
        # 4. Extract features using ONLY the bounding boxes found by MediaPipe
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([my_face_encoding], face_encoding)
            
            name = "Unknown"
            color = (0, 0, 255) 

            if matches[0]:
                name = "Me"
                color = (0, 255, 0)
                
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    out.write(frame)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()