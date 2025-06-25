# Image-Classification-On-Cifar-100
Implementations in pytorch of different network architectures for image classification on the Cifar-100 dataset 

### Simple CNN

```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```

Input Layer
Convolutional layer
Convolutional layer
Convolutional layer 
Flatten
Fully Connected Layer 1
Fully connected Layer 2

Uses Batch normalisation + ReLU in hidden layers 
Uses Maxpooling for downsising 

Accuracy of the network on the 10000 test images after 20 epochs: 22.43 % 

