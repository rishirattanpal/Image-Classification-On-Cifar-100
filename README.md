# Image-Classification-On-Cifar-100
Implementations in pytorch of different network architectures for image classification on the Cifar-100 dataset 

## Simple CNN
Network that serves the foundation for future models I make

Key features:
- Simple VGG-style architecture
- Batch norm + ReLU 
- Dropout for regulisation  

```mermaid
graph TD
    A[Input 3x32x32] --> B[Conv3x3: 3→64]
    B --> C[BatchNorm+ReLU]
    C --> D[MaxPool 2x2]
    D --> E[Conv3x3: 64→128]
    E --> F[BatchNorm+ReLU]
    F --> G[MaxPool 2x2]
    G --> H[Conv3x3: 128→256]
    H --> I[BatchNorm+ReLU]
    I --> J[MaxPool 2x2]
    J --> K[Flatten 256x4x4→4096]
    K --> L[FC: 4096→512]
    L --> M[Dropout 0.5]
    M --> N[ReLU]
    N --> O[FC: 512→100]
```



Accuracy of the network on the 10000 test images after 20 epochs: 22.43 % 


## ResNet 

Implementation of ResNet 

Key improvements:
- implemented a Resdiual block that utilises skip connections and progressive downsampling 
- Deeper network
- Kaiming initalisation
- More epochs

```mermaid
graph TD
    A[Input 3x32x32] --> B[Conv7x7: 3→64, stride=2]
    B --> C[BatchNorm+ReLU]
    C --> D[MaxPool 3x3, stride=2]
    D --> E[ResBlock x3: 64→64]
    E --> F[ResBlock x4: 64→128, stride=2]
    F --> G[ResBlock x6: 128→256, stride=2]
    G --> H[ResBlock x3: 256→512, stride=2]
    H --> I[GlobalAvgPool]
    I --> J[FC: 512→100]
```

Accuracy of the network on the 10000 test images after 60 epochs: 44.35 % 


## WideResNet

Implementation of WideResNet 

Key improvements:
- Implemented a Wide Residual block 
- Wider not deeper network
- Data regulization (CutMix/MixUp, RandomCrop, normalise to SD/mean, +more)

```mermaid
graph TD
    A[Input 3x32x32] --> B[Conv3x3: 3→16]
    B --> C[BatchNorm+ReLU]
    C --> D[WideBlock x3: 16→16×6]
    D --> E[WideBlock x4: 96→32×6, stride=2]
    E --> F[WideBlock x6: 192→64×6, stride=2]
    F --> G[WideBlock x3: 384→128×6, stride=2]
    G --> H[GlobalAvgPool]
    H --> I[FC: 768→100]
```

Accuracy of the network on the 10000 test images after 60 epochs: 72.22 % 


