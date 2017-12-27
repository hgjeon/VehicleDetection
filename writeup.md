
# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/CarImage.png
[image2]: ./output_images/NonCarImage.png
[image3]: ./output_images/HogChannelCar.png
[image4]: ./output_images/HogChannelNonCar.png
[image5]: ./output_images/Slidingwindow1.png
[image6]: ./output_images/Slidingwindow2.png
[image7]: ./output_images/Slidingwindow3.png
[image8]: ./output_images/HeatmapImage.png
[image9]: ./output_images/LabelImage.png
[image10]: ./output_images/FinalImage.png

[image9]: ./examples/labels_map.png
[image10]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, and following parameters has been used in the final result.
- Channel: `ALL`, use all 3 color channels
- Color Space: Choose `YCrCb` among RGB, HSV, LUV, HLS, YUV, YCrCb options
- `pixels_per_cell`: (8, 8)
- `cells_per_block`: (2, 2)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The feature has been extracted combining following three feature set
- Binned color feature: `YCrCb`, resize to (32, 32)
- Color histograms: `YCrCb`, 32 Histogram bins
- HOG feature: `YCrCb`, `ALL` channels

Once feature has been extracted, then it has been normalized using:                 

```     
# Features of Car and Non-Car
X = np.vstack((car_features, notcar_features)).astype(np.float64)  
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```

I trained a linear SVM (Support Vector Machine) using LinearSVC function in sklearn:
```
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

##### Image Cropping
I decided to search window positions in bottom half of image with various scales.
```
Scale Factors: [1.5, 2.5]
y_start_stop_ratio: [0.55, 0.90]
```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]

Here's an example result showing the heatmap:
![alt text][image8]

Here is the output of scipy.ndimage.measurements.label() on the integrated heatmap:
![alt text][image9]

Here the resulting bounding boxes are drawn onto the original frame as car detection:
![alt text][image10]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

##### Moving Average Heat Map
I recorded the positions of positive detections in more than 20 consecutive frames of the video. From the positive detections I created a heatmap and then I applied thresholded to identify vehicle positions.

##### Hysteresis on Detected box
Once a car is detected, the detected box with weighted factor has been added when I calculate the threshold for the next frame.

##### Label
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### False detection
I spent weeks to eliminate the false detection issue using all different techniques. I still see one spot in the center of the road.

##### Better training
Here I used HOG, and Linear SVM for the training. The Convolutional Neural Network with YOLO could be used for better performance.
