# ImageClassifierUdacity
In this project, you'll train an image classifier to recognize different species of flowers.


## Dataset 
The dataset was used [Flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), it includes 102 of flower categories.
The data split into three sets: training, validation and testing.

## Build model
We used Vgg16 to build our model by freeze the last layers and add new linear layers followed by ReLu activation function and softmax to return probabilities of the predication classes.

**VGG16 Architecture**
![alt text](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)
