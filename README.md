# Image-Classification-On-Cifar-100
Implementations in pytorch of different network architectures for image classification on the Cifar-100 dataset 

### Simple CNN
Network that serves the foundation for future models I make

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

