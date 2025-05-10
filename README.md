# Human-Activity-Recognition
[https://arxiv.org/abs/2304.14499](https://arxiv.org/abs/2304.14499)
## Human Activity Recognition using Single Frame CNN
 
### Approach:
The **Single Frame CNN** takes **individual video frames** and extracts meaningful spatial features using a **Convolutional Neural Network**. These features are then used to classify the human activity in the frame. This approach eliminates the need for modeling temporal dependencies, often considered computationally expensive, while still achieving competitive accuracy. 

The model would perform image classification on single frames of the video to recognize the action being performed. The model would then generate a probability vector for each input frame, which would denote the probability of different activities being present in that frame. All the individual probabilities would then be averaged to get the final output probabilities vector. Rather than processing every frame in long videos, the model processes a few frames spread throughout the video, **reducing computational cost**.


### Key Highlights:
- **Efficiency**: This method achieves high accuracy with just a single frame per activity, reducing the need for complex sequence modeling.
- **High Accuracy**: Despite being a simpler approach, our Single Frame CNN model outperforms the other method we explored - ConvLSTM, in terms of classification accuracy.
  

### Try It Out:
You can view this model in Google Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oTTgNpTiXCVyb5KDwYj4dK1MuoH-bWaA)



## Human Activity Recognition using ConvLSTM

### Approach:
The **Convolutional Long Short-Term Memory Networks (ConvLSTM)** model captures **spatiotemporal dependencies** in videos by combining **Convolutional Neural Networks (CNNs)** to extract spatial features and **Long Short-Term Memory (LSTM)** networks to capture temporal relationships between consecutive frames. This architecture enables the model to recognize activities based on sequences of frames.

### Key Highlights:
- **Spatiotemporal Modeling**: The model effectively handles both spatial and temporal features, allowing it to recognize complex activities across multiple frames.
- **Complexity**: Although ConvLSTM achieves reasonable accuracy, it is computationally more expensive than single-frame models like CNN, and it was outperformed by the **Single Frame CNN** model.

### Try It Out:
You can view this model in Google Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tsy4fkfOWGwF0pqEhSPLI96YSi0put2-)


## Results:
Our experiments show that the **Single Frame CNN** achieves superior accuracy compared to **ConvLSTM**. This highlights the potential of using single-frame models for real-time HAR applications, with significantly less computational overhead.

<img src="https://github.com/user-attachments/assets/5d155179-a72d-4e24-b3a7-8a0f67813d7e" width="300" height="100"/>

## Languages & Libraries/Frameworks Used

Python, OpenCV, TensorFlow, Scikit-learn, Numpy and Matplotlib
