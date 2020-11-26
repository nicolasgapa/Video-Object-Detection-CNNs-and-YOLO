"""

Object Detection using Transfer Learning (DarkNet)

Nicolas Gachancipa
Embry-Riddle Aeronautical University

Inputs:
    img_or_vid(str): 'img' for images, 'vid' for videos.
    input_video(str): Directory to input file.
    output_video(str): Directory to output file.
    weights_file(str): Directory to pre-trained DarkNet weights.
    cfg_file(str): Directory to CNN architecture (cfg format).
    labels_file(str): Directory to file containing the labels.
    save_frames(boolean): True if you want to save the frames of your video
                          apart (in a .jpg format).
    n_objects(int): If save_frames is set to True, only frames where more than 
                    n_objects have been detected will be saved.
    labels_output(str): Directory to save the Numpy array containing the 
                        labels of the detected objects (per frame) in a video.
    rotate(boolean): Rotates the image or video by 90 degrees, if set to True.
                    
Output:
    Processed image or video, with the corresponding bounding boxes. Also, 
    for videos, a Numpy (.npy) file containing an array with the detected 
    objects per frame.
    
References:
    Some parts of this code were based on a previous implementation 
    of YOLO video detection by Adrian Rosebrock. The original code can be 
    found in the following directory:
    https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
"""
# Imports.
import cv2
from cv2 import (VideoCapture as VC, CAP_PROP_FRAME_COUNT as n_frames, 
                 VideoWriter as CW, CAP_PROP_FPS as n_fps)
from cv2.dnn import readNetFromDarknet as DarkNet
from image_function import image_process
import math as mt
import matplotlib.pyplot as plt
import numpy as np

# Inputs.
img_or_vid = 'img'
input_file = 'Input/Videos/highway_video.mp4'
output_file = 'Output/Videos/highway_video.avi'
weights_file = 'yolov3_darknet.weights'
cfg_file = 'yolov3_darknet.cfg'
labels_file = 'coco.csv'
rotate = False

# Video Inputs (ignore if you are processing an image).
save_frames = False
n_objects = 3
labels_output = 'Output/Labels/day_1_data.npy'

# Load the labels from COCO.
with open(labels_file, 'r') as lines:
  coco_labels = lines.readlines()[0].split(',')
  
# Import the DarkNet model, and load the pre-trained weights.
net = DarkNet(cfg_file, weights_file)

# Get the colors tha will be used to create the bounding boxes.
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(coco_labels), 3))
  
# Part 1: Video processing.
if img_or_vid == 'vid':
    
    # Use OpenCV to load the video and obtain the number of frames.
    video = VC(input_file)
    number_of_frames = int(video.get(n_frames))
    fps = video.get(n_fps)

    # Create a new video.
    _, first_frame = video.read()
    video_shape = first_frame.shape
    new_video = CW(output_file, cv2.VideoWriter_fourcc('M','J','P','G'), mt.floor(fps), 
                  (video_shape[1], video_shape[0]), isColor = True)   
    
    # Process the video.
    classes = []
    count = 0
    n = 1 # Change this value if you want to process non-sequential frame.
          # For example, n = 12 will process the video every 12 frames. 
    while video.isOpened():
    
        # Print the current frame number, and progress. 
        progress = round(count*100/number_of_frames, 2)
        print('Frame count: ', count, ', Progress: ', progress, '%')
        if progress >= 100:
          break
    
        # Read the next frame.
        _, frame = video.read()
    
        # Process the current frame using the image_process function.
        # This returns a frame with the identified objects and their 
        # respective rectangles. It also returns the indexes of the classes
        # that have been identified in the image.
        if rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
        frame, idxs = image_process(frame, net, coco_labels, colors)
        classes.append(idxs)
    
        # Save frame to the new video.
        new_video.write(frame)
    
        # Save frames (only if more than n_objects were identified).
        if save_frames and len(idxs) > n_objects:
          print('Saving frame: ', count)
          cv2.imwrite('Output/Videos/frames/frame{}.jpg'.format(count), frame)
    
        # Set the count of the next frma to be processed.
        count += n
        video.set(1, count)
        
    # Compile new video.
    new_video.release()
    
    # Save the labels of the identified objects, for traffic processing 
    # analysis.
    np_array_classes = np.array(classes)
    np.save(labels_output, np_array_classes)
    
# Part 2: Image processing.
elif img_or_vid == 'img':
       
    # Show original image.
    frame = cv2.imread(input_file)
    
    # If the image is flipped, rotate it by uncommenting this line of code. 
    if rotate:
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
    
    # Show original image.
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Process the image.
    processed_frame, idxs = image_process(frame, net, coco_labels, colors)
    
    # Show processed image.
    plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Save the image.
    cv2.imwrite(output_file, frame)
    
    
