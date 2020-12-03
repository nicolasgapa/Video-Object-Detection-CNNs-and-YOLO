"""

Traffic Pattern Modelling using CNNs

Nicolas Gachancipa
Embry-Riddle Aeronautical University

Please refer to the attached PDF for more information about this algorithm.

Inputs:
    initial_time(time): Start time of the recording.
    n(int): Moving-average window (in seconds). 
    label(str): Object type. Examples: 'person', 'bicycle', 'car', 'motorbike'
    directory(str): Directory to the .npy file containing the labels.
                    
Output:
    Plot (showing the traffic patterns).

"""
# Imports.
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import timedelta, time, date, datetime

# Inputs.
initial_time  = time(8, 0, 35)
n = 600
label = 'all'
directory = 'Output//Labels//traffic_data.npy'

# Import labels array.
full_array = np.load(directory, allow_pickle=True)
    
# Identify objects. 
times_array = []
obj_ct_array = np.zeros((full_array.shape[0], 60))
for i, element in enumerate(full_array):
    if len(element) > 0:
        cts = [element.count(i) for i in range(0, 60)]
        obj_ct_array[i] = cts
    times_array.append(i)
    
# Define labels.
with open('coco.csv', 'r') as lines:
  coco_labels = lines.readlines()[0].split(',')
    
# Moving average.
if label == 'all':
    df = obj_ct_array.sum(axis=1)
    series = pd.Series(df)
else:
    series = pd.Series(obj_ct_array[:, coco_labels.index(label)])
moving_average = series.rolling(n).mean()

# Plot.
dt_initial = datetime.combine(date.today(), initial_time)
times = [dt_initial + timedelta(seconds=i) for i in times_array]
plt.plot(times, moving_average, '-')
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.ylabel('Average objects per frame. \n Class: {}.'.format(label))
plt.xlabel('Time of the day')
plt.show()
