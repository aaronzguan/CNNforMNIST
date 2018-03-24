# CNNforMNIST
A CNN for MNIST Dataset, the details could be found at http://www.aaronguan.com/CNNforMNIST.html

## Structure

<img src="https://github.com/aaronzguan/CNNforMNIST/blob/master/stucture.png" width="500"/>

<img src="https://github.com/aaronzguan/CNNforMNIST/blob/master/stucture_graph.png" width="500"/>

## How to read

[mainProcess.py](/without_visulization/mainProcess.py): main processing for training and testing.

[functionDef.py](/without_visulization/functionDef.py): simplifying the function: 2-D convolution, maximum pooling, bias and weight.

[CNNLayers.py](/without_visulization/CNNLayers.py): defines the struture of the CNN: 1 input layer, 2 convolution layers and 2 fully-connected layers.

[lossFunction.py](/without_visulization/lossFunction.py): defines the loss function and the accuracy of the CNN.

## Tensorboard visualization

[main.py](/main.py) embedded the tesorboard visulization function.

[log](/log) contains the output log of the program

[summary](/summary) contains the tensorboad visulization file

### How to visualize?

    tensorboard --logdir= "path/to/log-directory"
