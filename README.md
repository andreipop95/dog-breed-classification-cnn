# Dog Breed Classifier CNN
Dog breed classification with convolutional neural networks in pytorch

This project has been done as a part of the [Udacity Deep Learning Nanodegree](https://eu.udacity.com/course/deep-learning-nanodegree--nd101)


## Project's objective
The objective of the project was to train a CNN that can predict the breed of a dog from an image. As extension to this, if the image contains a human face, then the face is detected, and the resmbling dog breed is specified. 


## About current version of the project 
In order to solve the classification problem, multiple versions of [Convolutional Neural Nerworks](https://en.wikipedia.org/wiki/Convolutional_neural_network) have been used. The follwing steps have been performed:

1. **Build a convolutional neural network from scratch**
   
   For this task, the objective has been to get a network architecture that would predict the dog breed with an accuracy of at least 10% accuracy. 

   The architecture that has been used is presented below, and it performs with **16% accuracy**, after 20 epochs of training. 

   **_Model architecture_**:
![alt text](https://github.com/andreipop95/dog-breed-classification-cnn/blob/master/model_scratch.png)

   **Convolutional layers** - the number of convolutional layers used is 5, after trying multiple configurations. The kernel size used is 3. Relu has been used as activation function for each convolutional layer

   nn.Conv2d(3, 16, 3): 3 -> 16 channels<br/>
   nn.Conv2d(16, 32, 3): 16 -> 32 channels<br />
   nn.Conv2d(32, 64, 3): 32 -> 64 channels<br />
   nn.Conv2d(64, 128, 3): 64 -> 128 channels<br />
   nn.Conv2d(128, 256, 3): 128 -> 256 channels<br />

   **Pooling layer** - the pooling layer is applied after each convolutional layer. The parameters set for this layer are kernel_size = 2 and stride = 2

   **Dropout layer** - applied before the last layer - added in order to prevent overfitting.
Fully connected layer - used in order to predict the probability that an input belongs to a certain class. The total number of classes is 13.

   **Batch normalization layers** - after reading about how to improve prediction accuracy of the network, I have come to the conclusion that batch normalization would be a good improvement
   
2. **Build a network using transfer learning**

   The second network model has been built using [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning). The pretrained network used for this task is [VGG16](https://github.com/hadikazemi/Machine-Learning/blob/master/PyTorch/tutorial/vgg16.py).
   The objective for this step was to build a model that would give an accuracy of at least 60% when performing classification on test data.
   After training for 20 epochs with the follwing architecture an **accuracy of 79%** was obtained during classification.
   
   **_Model architecture_**
   
![alt text](https://github.com/andreipop95/dog-breed-classification-cnn/blob/master/model_transfer.png)

   For this model, the last layer has been adapted such that if would fit the domain of the classification problem. 

## Resources used
The dataset used for this classification problem has been provieded by the Udacity course, and it can be found at the links below:
* [Dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* [Human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

## How to run the project
The prerequisites in order to set the project on a local machine are:
* [Github](https://gist.github.com/derhuerst/1b15ff4652a867391f03)
* [Anaconda and python](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)

1. Git clone the repository
   
   `git clone https://github.com/andreipop95/dog-breed-classification-cnn.git`

2. Create a conda environment using python 3.5. As an alternative for package management, pip can be used

   `conda create --name <your_project_name> python=3.5`
   `source activate <your_project_name>`

3. Install needed packages
   
   `conda install numpy cv2 matplotlib torch torchvision PIL`
   
   If other packages are needed, you can add them by running the command `conda install <package_name>`
   
   
4. Install jupyter notebook
   
   `conda install jupyter notebook`
 
5. Lunch the notebook
   
   `jupyter notebook dog_app.ipynb`
