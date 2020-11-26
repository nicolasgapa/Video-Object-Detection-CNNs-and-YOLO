# Traffic Pattern Modelling using Convolutional Neural Networks (CNNs)

<p float="left" align="center">
  <img src="/Output/Images/highway_2_output.jpg" width="400"  />
  ![Alt Text](/Output/Videos/security_camera_output.gif)
  <img src="/Output/Images/highway_3_output.jpg" width="400"  />
</p>

This project is divided in three parts. 

1. Classification network (classification_network.py).

          The file can be used to train a convolutional neural network for a classification task.
          The model is uses an Inception network (Inception V1 or GoogLeNet). The model
          uses TensorFlow Keras, and was trained using Google Colab (See the attached 
          Colab Notebook for reference - classification_model.ipynb).

          In order for this file to work, you must add training and validation images
          to the respective folders in the Dataset folder. Those were not added to the 
          Github repository due to size constraints. However, training images can be 
          downloaded from the COCO dataset or any other open-source image datatset.
   
2. Object detection in images and videos using DarkNet and YOLO (object_detection.py).

          The object detection model is capable of identifying objects in an image or video, 
          and drawing bounding boxes around them. In order to use this model, you must define
          the directory to the input file (a jpg, mp4, png file, etc). You must also define
          the output directory, where the processed file will be saved. Some examples of 
          processed files (with bounding boxes) can be found in the Output folder of this
          repository. Moreover, the object_detection.py file saves a Numpy file (.npy)
          containing the labels of the objects detected in a video (it returns a list of 
          the objects found per video frame). 

          Note: In order for this model to work properly, the pre-trained weights of the DarkNet
          model must be downloaded and placed inn= the same directory as the object_detection.py
          file. The weights were not uploaded to Github due to size contraints. However, they can be
          downloaded from the following link (download the Yolov3-416 weights file):

          https://pjreddie.com/darknet/yolo/

   
3. Traffic Pattern Modelling (traffic_pattern_modelling.py)

          The traffic_pattern_modelling.py file uses the labels obtained from part 2 and
          plots traffic patterns for a specified object type (cars, people, motorcycle). The model
          was tested using security camera recordings of a residential street, and with short recordings 
          of a highway.
   

<p float="left" align="center">
  <img src="/Output/Images/dog_output.jpg" width="200"  />
  <img src="/Output/Images/krakow_output.jpg" width="200"  />
  <img src="/Output/Images/university_output.jpg" width="400"  />
</p>
   
