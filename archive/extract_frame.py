import cv2

cap = cv2.VideoCapture("driving_720p.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)  # grab frame 500, mid-video
success, frame = cap.read()
if success:
    cv2.imwrite("test_frame.jpg", frame)
    print("Saved test_frame.jpg")
else:
    print("Failed to read frame")
cap.release()