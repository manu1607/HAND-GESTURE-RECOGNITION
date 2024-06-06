

# Hand Gesture Recognition using Convolutional Neural Networks (CNN)

## Objective
The objective of this mini-project is to develop CNN models capable of accurately classifying sign language letters based on image inputs. By training neural networks on a dataset of sign language gestures, the models will aim to recognize and classify hand gestures into their corresponding letters.

## Dataset
The dataset consists of grayscale images of hand gestures representing sign language letters. Each image is of size 28x28 pixels, and the dataset includes labeled examples for training and testing.

### Features
- Input Image: Grayscale images of hand gestures representing sign language letters.
- Target Variable: The target variable is the class label corresponding to the sign language letter depicted in the input image.

## Model Architecture
The CNN architecture consists of the following components:
1. First Convolutional Layer: Applies a set of learnable filters (kernels) to the input data to extract features.
2. Second Convolutional Layer: Applies another set of learnable filters to capture more abstract and higher-level features.
3. Activation Function: ReLU activation function is applied after each convolutional layer.
4. Max Pooling Layer: Down-samples the feature maps to reduce spatial dimensions.
5. Dropout Layer: Helps prevent overfitting by introducing randomness into the network.
6. Output Layer: Consists of 26 nodes for each class (A-Z).

### Model Visualization
![Model Architecture](model_architecture.png)

## Model Training
- Compilation: The model is compiled using the Adam optimizer.
- Training: The model is trained for 10 epochs.
- Evaluation: Changes in loss and testing accuracy for each epoch are monitored.

## Results
Analyzing variations in loss and testing accuracy over epochs helps assess model performance.

### Observations on Overfitting
Overfitting issues may arise if the model learns to memorize the training data instead of generalizing well to unseen data. Dropout layers help prevent overfitting by introducing randomness into the network during training.

### Handling Overfitting
Adding another dropout layer (e.g., with rate 0.3) after the convolutional layers may further reduce overfitting by introducing more randomness into the network.

### Share Structure and Invariance Property
The CNN architecture inherently leverages the share structure and invariance property in image classification tasks. By stacking convolutional layers followed by max pooling layers, the network learns to extract increasingly abstract and hierarchical features from the input data while being invariant to translations and distortions.




## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/nn.html)
- [PyTorch BCELoss Documentation](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss)
- [How to Save and Load Models in PyTorch](https://wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE)
- [Handwritten Character Recognition in Assyrian Language using CNN](https://www.researchgate.net/publication/379382573_HANDWRITTEN_CHARACTER_RECOGNITION_IN_ASSYRIAN_LANGUAGE_USING_CONVOLUTIONAL_NEURAL_NETWORK)


