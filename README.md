# CNNforMNIST
A CNN for MNIST Dataset, the details could be found at http://www.aaronguan.com/CNNforMNIST.html

## Structure

<img src="https://github.com/aaronzguan/CNNforMNIST/blob/master/stucture.png" style="width: 200px;"/>

![image](https://github.com/aaronzguan/CNNforMNIST/blob/master/stucture_graph.png)

## How to read

[mainProcess.py](/mainProcess.py): main processing for training and testing.

[functionDef.py](/functionDef.py): simplifying the function: 2-D convolution, maximum pooling, bias and weight.

[CNNLayers.py](/CNNLayers.py): defines the struture of the CNN: 1 input layer, 2 convolution layers and 2 fully-connected layers.

[lossFunction.py](/lossFunction.py): defines the loss function and the accuracy of the CNN.
