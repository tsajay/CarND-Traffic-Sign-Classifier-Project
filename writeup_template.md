#**Traffic Sign Recognition** 



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data_vis.jpg "Visualization"
[image2]: ./original_sample.jpg "Original image"
[image3]: ./gray_sample.jpg "Grayed out normalized image"
[image4]: ./img1.jpg "Road work"
[image5]: ./img2.jpg "Right-of-way at the next intersection"
[image6]: ./img3.jpg "Stop"
[image7]: ./img4.jpg "Speed limit (60km/h)"
[image8]: ./img5.jpg "Yield"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tsajay/CarND-Traffic-Sign-Classifier-Project/)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 original images. With data augmentation, my training set had a total of 69598 images. (more below)
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Visualization][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The first step in my processing pipeline is to augment the existing images with random shifts of 1-pixel in one of the four directions - up, down, left, or right. I did this by padding the image with zeros in the edges. Note that this step was added after I found my existing dataset to be a bit small to consistently achieve the desired accuracy. Also, this is one of the techniques I read about in the landmark paper by [Hinton, et.al](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). That paper rotates images laterally, but for traffic signs, you need to recognize text, and probably signs like "turn left". Hence rotation is not a technique that can be applied here. Shifts should still preserve most of the data required to classify the image.

Next, I converted the image to grayscale. Once I obtain the mean and standard deviation of the pixel values across all images, I normalized each image. This is another technique from the paper mentioned above.

Normalization is a required step for most machine learning algorithms to accelerate convergence. 

![Original Sample][image2]
![Gray Sample][image3]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Please see the above section. I did augment my data with shifts in the images by 1-pixel. This was after I found that 35K images are not enough to consistently converge to the desired accuracy.

I chose LeNet as the basis of my neural network for training with the following modifications to the network. I added one extra convolution layer and one extra fully-connected layer. I felt this is required since LeNet was designed to recognize digits, so the neural network is mostly required to detect edges and curves, but not more complex shapes as required for the traffic signs dataset. Also, the rationale for choosing LeNet was that the 32x32 images in this set matched closely with what LeNet was designed for (28x28 images). I changed a few parameters in my network in the fully-connected layers to accomodate for native 32x32 image sizes.


The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 5x5x16 	|
| Fully connected		| 400x240        									|
| RELU					|												|
| Fully connected		| 240x160        									|
| RELU					|												|
| Fully connected		| 160x96        									|
| RELU					|												|
| Fully connected		| 96 x n_classes        									|
| Softmax				| Probabilities of various classes.        									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

After various experimentation, batch size of 128 still was useful. The general trend was that the training accuracy increased with smaller batch sizes, but it did not affect test accuracy by much. The downside of smaller batch sizes was the longer training times as well. 

Training for 40 epochs was enough for convergence in all cases. My network consistently achieved derised accuracy for test-set over 25 runs. Note that the input is randomzied. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 95.8%
* validation set accuracy of 95.8% for the final epoch (started from 90.8% accuracy for the first epoch)
* test set accuracy of 94.0%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
Raw LeNet did not have enough discriminatory power for complex images. I had to add more layers in the convolution output. I also added more fully connected layers to be able to detect more complex shapes than numbers. I had to adjust the parameters to account for a 32x32 image rather than 28x28 image. For example, some of the FC layer parameters in original LeNet were determined by 28(width of image)x3 = 84, I had to change them to 96.
* Which parameters were tuned? How were they adjusted and why?
In addition, I changed the number of epochs to 40 for consistent training convergence. My best guess is that this helped the training get over some local minima. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
With random augmentations of the image data, we are providing a slightly different dataset each time to the training network. Still, my network always converges, and has an accuracy of over 93% on the test dataset. I can conclude with fair degree of confidence that the parameters chosen for the network are working. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Traffic signals   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Stop					| Stop											|
| Speed limit (60km/h)	      		| Speed limit (60km/h)					 				|
| Yield			| Yield      							|


UPDATE: This is my second submission, mostly for my program to print the softmax probabilities. The model was able to correctly guess 80% of the downloaded signs. I downloaded 10 other images and my network was able to correctly guess all of them. I am not changing the test image from my first submission just for consistency. Also, here's a reason why the network is converging to a traffic sign. The training set is in monochromatic gray color. There's a fairly large circular blob in the image, which can confuse the network into thinking that it's a traffic sign. Nice learning experience (for my network and for me).

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

First image:
Top 5 predictions        : Traffic signals, General caution, Right-of-way at the next intersection, Go straight or left, Turn right ahead
Prediction (Idx: 26)     : Traffic signals 


Second image:
Top 5 predictions        : Right-of-way at the next intersection, Pedestrians, Roundabout mandatory, Speed limit (20km/h), Speed limit (30km/h)
Prediction (Idx: 11)     : Right-of-way at the next intersection 

Third image:
Top 5 predictions        : Stop, Yield, No entry, Speed limit (30km/h), Bumpy road
Prediction (Idx: 14)     : Stop 

Fourth image:
Top 5 predictions        : Speed limit (60km/h), Dangerous curve to the left, Wild animals crossing, Speed limit (50km/h), Slippery road
Prediction (Idx:  3)     : Speed limit (60km/h) 

Fifth image:
Top 5 predictions        : Yield, Speed limit (20km/h), Speed limit (30km/h), Speed limit (50km/h), Speed limit (60km/h)
Prediction (Idx: 13)     : Yield 
