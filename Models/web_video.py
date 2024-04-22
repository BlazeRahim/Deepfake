import cv2
import os

def frames_generate(video_path, output_base_folder,random_filename):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    desired_width = 200
    desired_height = 200
    
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    vid = cv2.VideoCapture(video_path)
    currentframe = 0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        success, frame = vid.read()
        if not success:
            break
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                expansion_factor = 0
                expanded_x = max(0, x - int(w * expansion_factor))
                expanded_y = max(0, y - int(h * expansion_factor))
                expanded_w = min(frame.shape[1] - 1, x + w + int(w * expansion_factor)) - expanded_x
                expanded_h = min(frame.shape[0] - 1, y + h + int(h * expansion_factor)) - expanded_y
                face_frame = frame[expanded_y:expanded_y + expanded_h, expanded_x:expanded_x + expanded_w]
                face_frame = cv2.resize(face_frame, (desired_width, desired_height))
                name = os.path.join(output_base_folder, f'{random_filename}_frame{currentframe}.jpg')
                cv2.imwrite(name, face_frame)
                currentframe += 1
    vid.release()
    cv2.destroyAllWindows()
    return total_frames