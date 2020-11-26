"""
Created on Thu Oct 22 20:02:02 2020

Nicolas Gachancipa

Some parts of this code were based on a previous implementation 
by Adrian Rosebrock. The original code can be found in the following directory:
https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

"""
# Import the necessary packages
import numpy as np
import cv2
from cv2.dnn import blobFromImage, NMSBoxes
import itertools
import pandas as pd

# Functions for image processing.
def image_process(frame, network, labels, colors, size = 416):
    
    # Extract the original dimensions of the image.
    H, W = frame.shape[0], frame.shape[1]
    
    # Normalize the image, and reshape it to the given size.
    network.setInput(blobFromImage(frame, 1/255, (size, size), swapRB=True))
    
    # Identify the last layers of the Network. The DarkNet network has
    # three output layers: YOLO 82, YOLO 94, and YOLO 106. Flow the image
    # through the CNN and save the three outputs to a layer called last_layers.
    last_layers = []
    for i in network.forward(['yolo_82', 'yolo_94', 'yolo_106']):
        last_layers.append(i)
    output = np.array(list(itertools.chain.from_iterable(last_layers)))
    
    # Save the output to a pandas dataframe.
    full_output = pd.DataFrame(data=output)
    
    # Obtain the scores of the CNN, which are the 5th through 85th columns.
    # Convert the results to a data_frame, and filter to obtain only the 
    # objects that have a high confidence, above 60%.
    scores = output[:, 5:]
    df = pd.DataFrame(data=scores)
    filtered_df = df > 0.6
    filtered_output = full_output[filtered_df.isin([True]).any(axis=1)]
    
    ############################################################
    
    # The following section of the code was developed based on a previous 
    # implementation by Adrian Rosebrock. The original code can be found 
    # in the following directory:
    # https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
    
    # Identify the boxes, their probabilities, and the identified classes.
    boxes, probabilities, classes = [], [], []
    for _, row in filtered_output.iterrows():
        row = np.array(row)
        scores = row[5:]
        idx = np.argmax(scores)
        cx, cy, w, h = row[0:4] * np.array([W, H, W, H])
        boxes.append([cx - (w / 2), cy - (h / 2), w, h])
        probabilities.append(float(scores[idx]))
        classes.append(idx)

	# Apply non-maxima suppression. CV2 returns the best bounding boxes.
    idxs = NMSBoxes(boxes, probabilities, 0.5, 0.3).flatten()
    bounding_boxes = np.array(boxes)[[idxs], :4][0]

    # The labs array contains the indices of the labels that were detected.
    # This array is returned by the function, along with the updated image.
    labs = []
    
    # Draw rectangles and add text (extracted and slightly modified from 
    # Adrian Rosebrock's code).
    if len(bounding_boxes) > 0:
        
        # Process each of the boundig boxes returned by the non-maxima
        # supression step.
        for i, box in zip(idxs, bounding_boxes):
            
            # Draw a bounding box rectangle.
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            color = [int(c) for c in colors[classes[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Identify the label (to print on the image). Save the index of the
            # label to the labs array. 
            idx = classes[i]
            labs.append(idx)
            label = labels[idx]

            # Write the class and probability on top of the image.
            text = "{}: {:.2f}".format(label, probabilities[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    ############################################################
    

	# Frame
    return frame, labs