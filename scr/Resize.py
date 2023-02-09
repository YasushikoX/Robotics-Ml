import cv2
import os
import numpy as np

data_dir = r'C:\Users\yasus\Projects\Robotics-Ml\TrainData' # directory where the videos are stored
output_dir = r'C:\Users\yasus\Projects\Robotics-Ml\TrainData\resized' # directory where the output videos should be saved

# create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# loop through all the videos in the data directory
for video_name in os.listdir(data_dir):
    if video_name.endswith('.mp4'):
        print("Processing video:", video_name)

        # open the video using OpenCV
        cap = cv2.VideoCapture(os.path.join(data_dir, video_name))

        # check if the video was opened successfully
        if not cap.isOpened():
            print("Error opening video", video_name)
            continue

        # get the video frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Video fps:", fps)

        # get the video size
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print("Video size:", frame_width, "x", frame_height)

        # limit the fps to 24
        if fps > 24:
            fps = 24

        # resize the frames to meet the requirements of C3D
        frame_width = 256
        frame_height = 144

        # create an VideoWriter object to save the resized video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_dir, video_name), fourcc, fps, (frame_width, frame_height))

        # loop through the frames of the video
        while True:
            # read the next frame
            ret, frame = cap.read()

            # check if the frame was read successfully
            if not ret:
                break

            # resize the frame
            frame = cv2.resize(frame, (frame_width, frame_height),
                               interpolation=cv2.INTER_AREA)

            # write the resized frame to the output video
            out.write(frame)

        # release the video capture and VideoWriter
        cap.release()
        out.release()

        print("Done processing video:", video_name)

print("Finished processing all videos")
