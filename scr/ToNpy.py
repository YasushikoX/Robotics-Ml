import os
import cv2
import numpy as np

# Parameters
data_dir = r'C:\Users\yasus\Projects\Robotics-Ml\TrainData' # directory where the videos are stored
output_dir = r'C:\Users\yasus\Projects\Robotics-Ml\TrainData\npy_files' # directory where the npy files should be saved

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all the video files in the data directory
video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]


for video_file in video_files:

    cap = cv2.VideoCapture(os.path.join(data_dir, video_file))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))
    
    fc = 0
    ret = True
    while (fc < frame_count and ret):
        ret, frame = cap.read()
        if ret:
            buf[fc] = frame
            fc += 1

    cap.release()
    
    # Save the video frames as a .npy file
    np.save(os.path.join(output_dir, os.path.splitext(video_file)[0] + '.npy'), buf)
    
print("All files converted to .npy files!")