# Traffic Sign Recognition

## Introduction

### Applying a Convolutional Neural Network to classify images from the german traffic signs system

---

**Goals**

The goals / steps of this project are the following:  
* Load the data set   
* Explore, summarize and visualize the data set  
* Design, train and test a model architecture  
* Use the model to make predictions on new images  
* Analyze the softmax probabilities of the new images  
* Summarize the results with a written report  

**Code**

The code for this project is contained in a jupyter notebook included in this repo `Traffic_Sign_Classifier.ipynb` 

For esy review you may open the file `Traffic_Sign_Classifier.md` in your browser from this github page, which is a markdown version of the code.


##Dataset Exploration

The data set consists of images retrieved from the German traffic system. The following is a summary of the dataset:

Number of training examples = 34799  
Number of testing examples = 12630  
Image data shape = (32, 32, 3)  
Number of classes = 43  

Images are sized to 32 by 32 pixels and are full RGB colour. The images are classified into 43 classes represented by numeric indices, which correpond to a class desription.

The following are the most frquent image classes in the set:  
Speed limit (50km/h) => 2010 examples  
Speed limit (30km/h) => 1980 examples    
Yield => 1920   examples  
Priority road => 1890   examples  
Keep right => 1860   examples  

This is a histogram showing the distribution of images per class index.

![alt text][image1]

This is a sample image with corresponding class label

![alt text][image2]

The code for the summary and visualization can be found in the 2nd and 3rd code cell of the Jupyter Notebook.

##Design and Test a Model Architecture

###Preprocessing

Two steps were applied to images for processing. First, the images were converted to grayscale so we can deal only with one colour space.

X_train = np.dot(X_train, [0.299, 0.587, 0.114])

Then normalization was applied to the image data. This helps with allowing the algorithm to find the optimal point doing less searching in the function space.

X_train = (X_train - 128.0) / 128

In this case values were normilized from -1 to +1 with a mean of 0.

The full code for the preprocessing can be found in cell 4 of the Jupyter Notebook.

###Model Architercture

The problem of classifying traffic sign images presents certain properties that makes it well suited for convolutional neural networks approach. Namely, we have images with certain known patterns, and we know that the patterns that make those signs are the same regardless of where in the image they appear. 

Because of these properties we can help the network by allowing it to share weights for the same identifiable elements in the image. Which is the basis for using stacks of convolutions over the image during the training process.

The model for this architecture is LeNet. The following table shows a summary of the layers of this model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		  | 32x32x1 Grayscale image   | 
| Convolution 6@28x28 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU					  | Activation function |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6 |
| Convolution 16@10x10| 1x1 stride, valid padding, outputs 10x10x16|
| RELU					  | Activation function |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x16 |
| Flatten             | 5x5x16 = 400|
| Fully connected	  | Input 400, Output 120|
| RELU					  | Activation function |
| Fully connected	  | Input 120, Output 84|
| RELU					  | Activation function |
| Fully connected	  | Input 84, Output 43 Logits|
 

The model basically combines a few layers of convolutions and pooling, followed by a few layers of fully connected nodes. 

Pooling is used in this model as an improvement in the way we reduce the feature space. Using a rather larger stride (i.e. 2) is an agressive method that drops a lot of information to downsample the image. With pooling we take a small stride of 1, but then combine the values of the convolutions in a particular neighborhood. In this model we use max pooling. This approach does not increases the number of parameters, so we don't risk of overfitting. It also often yields better models.

The activation function is the Rectified Linear Units, which allows for a non-linear model while keeping simplicity of linear functions and their derivatives.

After initial testing there didn't seem to be signs of overfitting. So Dropout was not considered at this time.

The code for the model can be found in cell 5. 

###Training Pipeline

A training pipeline is setup to use the model. First the parameters for Epochs, Batch Size and Learning Rate are defined as 12, 128 and 0.005 respectively. These values were found by trial and error running multiple training sessions.

The output of the network is processed using softmax to obtain probabilites of each class. Then cross entropy is used to calculate the loss operation. Finally the Adam Optimizer is used to minimize the loss during training.

In order to evaluate the performance of the model, we define the correct prediction as a vector that find out if our prediction matched the hot one encoding of the labels of the training set. The the overall accuracy is measured along the batches being processed.

Full code for the training pipeline can be found in cell 6.

###Training Session

The final results from training the model are as follows:

Training Accuracy: 0.984

Validation Accuracy: 0.924

Test Accuracy: 0.895

The first parameter that was tuned was the learning rate. We started with 0.001 with a validation accuracy of ~0.88 and started increasing the value until we reached 0.005. Any higher than that the model starts to underperform.

The second value was the number of epochs, which was optimal at 12. The accuracy stays between 0.98 and 0.93 for subsequent epochs. 

The batch size did not seem to have a significant effect on the results.

The training and evaluation code can be found in cells 7, 8 and 9

###Test on Web Images

The model was tested with the following images:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

These were preprocessd in the same way than the training images. My initial thoughts were that those watermarked images were going to be difficult to identify. 

After running the model on these images, all but one were correctly classified. The ones correctly classified had a great confidence with probability of 1.

The one that was not classified correctly was the last image seen above. This is the "no passing" image. The following are the top 5 classes (highest proabilities) pulled from the softmax applied to the logits for that image:

(Index => Label and Probailitiy)

![alt text][image9]

We can see that the fourth "guess" from the classifier was correct. However its probability was pretty low at less than 0.01. As it can be seen the confidence of the classifier here was not good for the top selection "No Vehicles" with prob of 0.5. The second and third top classes are kind of high comparatively. Specially "Speed Limit" with a prob of 0.4. So the classifier was uncertain about what this image is. Once reason could be the contract of the sample image from the web, compared to the ones in the training set.

The accuracy of the model in these test images was 0.8. Which seems lower than the test one, but the web test set is only 5 images.

The full code for the processing and analysis of these images can be found in cells 10 and 11.

[//]: # (Image References)

[image1]: ./report_img_1.png "Histogram"
[image2]: ./report_img_2.png "Sample Image"
[image3]: ./report_img_3.png "Sample Image"
[image4]: ./report_img_4.png "Sample Image"
[image5]: ./report_img_5.png "Sample Image"
[image6]: ./report_img_6.png "Sample Image"
[image7]: ./report_img_7.png "Sample Image"
[image8]: ./report_img_8.jpg "Sample Image"
[image9]: ./report_img_9.png "Sample Image"



