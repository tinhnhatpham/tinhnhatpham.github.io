---
layout: post # You can ommit this if you've set it as a default
title: The Importance of Initialization in Deep Learning
categories: [Deep Learning]
comments: true
---

# Table of contents

- [Introduction](#introduction)
- [Install Packages](#install-packages)
- [Dataset and Model Preparation](#dataset-and-model-preparation)
  - [Data Proccessing](#data-proccessing)
  - [Model Architechure](#model-architechure)
- [Initializer](#initializer)
  - [Zeros Initialization](#zeros-initialization)
  - [Large Random Values Initialization](#large-random-values-initialization)
  - [He Initialization](#he-initialization)
- [Conclusion](#conclusion)


# Introduction

In a previous post, we implemented [a deep neural network](https://tinhnhatpham.github.io/my_blogs/2022-12-07-implement-deep-neural-network). We have used some initialization methods for the network's weights $W$. To gain more intuition, we will go into detail about how they work.

In deep learning, the weights and biases of a neural network are typically initialized before training. This is an important step because the initialization of the weights can have a significant impact on the network's ability to learn and generalize.

There are several different approaches to initialization, and the choice of which one to use can depend on the specific architecture of the network, the type of data being used, and the optimization algorithm being employed.

We will reuse the neural network framework that we implemented, [click here](https://tinhnhatpham.github.io/my_blogs/2022-12-07-implement-deep-neural-network) for more detail. You can download the code [here](https://github.com/tinhnhatpham/deep_neural_network_implementation).

# Install Packages


```python
!cp /content/drive/MyDrive/Colab\ Notebooks/deep_neural_network_implementation/Initializer.py .
!cp /content/drive/MyDrive/Colab\ Notebooks/deep_neural_network_implementation/Activation.py .
!cp /content/drive/MyDrive/Colab\ Notebooks/deep_neural_network_implementation/Layer.py .
!cp /content/drive/MyDrive/Colab\ Notebooks/deep_neural_network_implementation/Loss.py .
!cp /content/drive/MyDrive/Colab\ Notebooks/deep_neural_network_implementation/NeuralNetwork.py .
```


```python
from Layer import *
from NeuralNetwork import *
```

# Dataset and Model Preparation

## Data Proccessing

Let generate the dataset and plot it.


```python
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np

# Generate 2d classification dataset
X_train, Y_train = sklearn.datasets.make_moons(n_samples=400, noise=0.2)
X_test, Y_test = sklearn.datasets.make_moons(n_samples=100, noise=0.2)

X_train = X_train.T
Y_train = np.reshape(Y_train, (1, -1))

X_test = X_test.T
Y_test = np.reshape(Y_test, (1, -1))

# scatter plot, dots colored by class value
plt.scatter(X_train[0, :], X_train[1, :], c=Y_train, s=15, cmap=plt.cm.Spectral);
```


    
![png](\assets\images\blogs\2022-12-26-the-importance-of-initialization-in-deep-learning-output_5_0.png)
    



```python
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
```

## Model Architechure

To test the ```Initializer```, we will use the neural network model with 3 layers.


```python
def get_model(initializer):
  model = NeuralNetwork()
  model.add(Dense(units=12, activation='relu', initializer=initializer))
  model.add(Dense(units=6, activation='relu', initializer=initializer))
  model.add(Dense(units=1, activation='sigmoid'))
  model.compile(loss='crossentropy')
  return model
```

# Initializer

In this section, we will train our neural network with ReLU activation with some initialization methods:

* Initialize $W$ with zeros.

* Initialize $W$ using large random values.

* Initialize $W$ using He method.

For bias $b$, we initialize to $0$ in all cases.

Let's create a ```random_normal``` function to generate weights $W$ from Standard Normal Distribution.


```python
def random_normal(self):
  weights = np.random.randn(*self.shape) * self.scale
  bias = np.zeros((self.shape[0], 1)) * self.scale
  return weights, bias

Initializer.random_normal = random_normal
```

## Zeros Initialization


```python
# Initialize weights with 
initializer = Initializer(scale=0)
dnn_model = get_model(initializer)

dnn_model.fit(X_train, Y_train, learning_rate=0.01, epochs=15000, print_log=True)
```

    Epochs: 0 - Cost: 0.6931271807599427
    Epochs: 1000 - Cost: 0.6931271807599427
    Epochs: 2000 - Cost: 0.6931271807599427
    Epochs: 3000 - Cost: 0.6931271807599427
    Epochs: 4000 - Cost: 0.6931271807599427
    Epochs: 5000 - Cost: 0.6931271807599427
    Epochs: 6000 - Cost: 0.6931271807599427
    Epochs: 7000 - Cost: 0.6931271807599427
    Epochs: 8000 - Cost: 0.6931271807599427
    Epochs: 9000 - Cost: 0.6931271807599427
    Epochs: 10000 - Cost: 0.6931271807599427
    Epochs: 11000 - Cost: 0.6931271807599427
    Epochs: 12000 - Cost: 0.6931271807599427
    Epochs: 13000 - Cost: 0.6931271807599427
    Epochs: 14000 - Cost: 0.6931271807599427
    Epochs: 14999 - Cost: 0.6931271807599427



```python
plt.plot(dnn_model.costs)
plt.ylabel('cost')
plt.xlabel('iterations (per thousands)')
plt.title("Learning rate =" + str(dnn_model.learning_rate))
plt.show()

print("Train", dnn_model.evaluate(X_train, Y_train))
print("Test", dnn_model.evaluate(X_test, Y_test))
```


    
![png](\assets\images\blogs\2022-12-26-the-importance-of-initialization-in-deep-learning-output_13_0.png)
    


    Train {'accuracy': 50.0}
    Test {'accuracy': 50.0}



```python
plot_decision_boundary(lambda x: dnn_model.predict(x.T) > 0.5, X_train, Y_train)
```


    
![png](\assets\images\blogs\2022-12-26-the-importance-of-initialization-in-deep-learning-output_14_0.png)
    


As you can see, when the weights and bias initialized to $0$, the model could not learn anything. The cost at every epochs is the same. Let's compute the cost of one example:

* Linear

  $$
    z = w*x + b = 0\text{ (since }w=0)
  $$

* ReLU Activation

  $$
    ReLU(z) = \text{max}(0,z) = 0
  $$

* Sigmoid Activation

  $$
    \sigma(z) = \widehat{y} = \frac{1}{1+e^{-z}} = \frac{1}{2}
  $$

* Loss Function

  $$
    \mathcal{L}_{(\widehat{y},y)} = -y\ln(\widehat{y})-(1-y)\ln(1-\widehat{y})\\
  $$

  * For $y=1$ and $\widehat{y}=0.5$:

  $$
    \mathcal{L}_{(\widehat{y},y)} = -(1)\ln(0.5)\approx 0.6931\\
  $$

  * For $y=0$ and $\widehat{y}=0.5$:

  $$
    \mathcal{L}_{(\widehat{y},y)} = -(1)\ln(0.5)\approx 0.6931\\
  $$

With zeros initialized, the $\widehat{y}$ is always $0.5$ regardless of $y$ value. The weights and bias became stuck at $0$ and learned nothing. That is why the model performs so poorly.

To solve this problem, we should initialize the network with random values so that the network's symmetry is broken and the parameters can begin to learn from the gradient descent algorithm.

## Large Random Values Initialization 


```python
initializer = Initializer(scale=10)
dnn_model = get_model(initializer)

dnn_model.fit(X_train, Y_train, learning_rate=0.01, epochs=15000, print_log=True)
```

    Epochs: 0 - Cost: 7.312751029251156
    Epochs: 1000 - Cost: 5.495101316884447
    Epochs: 2000 - Cost: 5.469131904930106
    Epochs: 3000 - Cost: 5.41242642347803
    Epochs: 4000 - Cost: 5.354672944310919
    Epochs: 5000 - Cost: 5.354224756932314
    Epochs: 6000 - Cost: 5.354068636349012
    Epochs: 7000 - Cost: 5.353951853501705
    Epochs: 8000 - Cost: 5.334756089007781
    Epochs: 9000 - Cost: 4.644965788626871
    Epochs: 10000 - Cost: 4.638886338370908
    Epochs: 11000 - Cost: 4.638203486132624
    Epochs: 12000 - Cost: 4.637757703730385
    Epochs: 13000 - Cost: 4.63741050583782
    Epochs: 14000 - Cost: 4.63711588197088
    Epochs: 14999 - Cost: 4.636862344557608



```python
plt.plot(dnn_model.costs)
plt.ylabel('cost')
plt.xlabel('iterations (per thousands)')
plt.title("Learning rate =" + str(dnn_model.learning_rate))
plt.show()

print("Train", dnn_model.evaluate(X_train, Y_train))
print("Test", dnn_model.evaluate(X_test, Y_test))
```


    
![png](\assets\images\blogs\2022-12-26-the-importance-of-initialization-in-deep-learning-output_18_0.png)
    


    Train {'accuracy': 85.99297141360952}
    Test {'accuracy': 82.9904410614862}



```python
plot_decision_boundary(lambda x: dnn_model.predict(x.T) > 0.5, X_train, Y_train)
```


    
![png](\assets\images\blogs\2022-12-26-the-importance-of-initialization-in-deep-learning-output_19_0.png)
    


Using random values for initialization allows the model to break symmetry, allowing us to see the network learn and the cost begin to decrease. But there are many things to consider:

* At the begining, the cost is too high. The reason is that we initilized with a large values. It makes the sigmoid activation ouput in the last layer to be close to the extremes (near $0$ or $1$), resulting in high loss in some cases.

* Initilizing with small random values can make it better. And in sigmoid activation, initialize random values from Gausian Distribution (the bell curve) might help avoiding close to the extremes than from Uniform Distribution.

## He Initialization

The He initialization method is a variant of the "Xavier initialization" method, which is designed to help prevent the "exploding gradients" problem that can occur in deep neural networks with rectified linear unit (ReLU) activation functions. The He initialization method is specifically designed for use with ReLU activation functions and has been shown to work well in practice for deep networks with a large number of layers.

To use the He initialization method, the weights of the network are initialized using a normal distribution with a mean of 0 and a standard deviation of sqrt(2/n), where n is the number of units in the layer being initialized. This initialization method has been found to work well in practice and is a popular choice for initializing the weights of deep neural networks.

* The weights $W$ is draw from truncated normal distribution:

  $$
    W_l \sim N\left(0, \sqrt{\frac{2}{n_{l-1}}}\right)\\
    \begin{cases}
      l&\text{layer }l^{th}\\
      n_{l-1}&\text{ number of units in previous layer}
    \end{cases}
  $$

* To achieve that, we can draw $W$ from standard normal distribution $N(0,1)$, and multiply it by $\sqrt{\frac{2}{n_{l-1}}}$.


```python
def he(self):
  scale = np.sqrt(2/self.shape[1])
  weights = np.random.randn(*self.shape) * scale
  bias = np.zeros((self.shape[0], 1)) * scale
  return weights, bias

Initializer.he = he
```


```python
dnn_model = get_model(initializer="he")

dnn_model.fit(X_train, Y_train, learning_rate=0.01, epochs=15000, print_log=True)

print("Train", dnn_model.evaluate(X_train, Y_train))
print("Test", dnn_model.evaluate(X_test, Y_test))
```

    Epochs: 0 - Cost: 0.6978053289268477
    Epochs: 1000 - Cost: 0.29883193419089327
    Epochs: 2000 - Cost: 0.2600368674263979
    Epochs: 3000 - Cost: 0.2323093033150798
    Epochs: 4000 - Cost: 0.20438349934048824
    Epochs: 5000 - Cost: 0.1736171657060345
    Epochs: 6000 - Cost: 0.14281607242627897
    Epochs: 7000 - Cost: 0.11815542629436156
    Epochs: 8000 - Cost: 0.10164325232730922
    Epochs: 9000 - Cost: 0.09028379017289069
    Epochs: 10000 - Cost: 0.0829836030618814
    Epochs: 11000 - Cost: 0.0779985623174174
    Epochs: 12000 - Cost: 0.07461697224492872
    Epochs: 13000 - Cost: 0.07223560554603126
    Epochs: 14000 - Cost: 0.07057944514301777
    Epochs: 14999 - Cost: 0.06929693689796101
    Train {'accuracy': 95.39460913104666}
    Test {'accuracy': 94.65618414215214}



```python
plt.plot(dnn_model.costs)
plt.ylabel('cost')
plt.xlabel('iterations (per thousands)')
plt.title("Learning rate =" + str(dnn_model.learning_rate))
plt.show()

print("Train", dnn_model.evaluate(X_train, Y_train))
print("Test", dnn_model.evaluate(X_test, Y_test))
```


    
![png](\assets\images\blogs\2022-12-26-the-importance-of-initialization-in-deep-learning-output_24_0.png)
    


    Train {'accuracy': 95.43587328241739}
    Test {'accuracy': 94.00148816395938}



```python
plot_decision_boundary(lambda x: dnn_model.predict(x.T) > 0.5, X_train, Y_train)
```


    
![png](\assets\images\blogs\2022-12-26-the-importance-of-initialization-in-deep-learning-output_25_0.png)
    


He initialization works really well when applied to the network with ReLU activation. The model prediction achieves 95% accuracy on training set as well as 94% on test set.

# Conclusion

* Poor initialization can result in vanishing/exploding gradients, slowing down the optimization algorithm.

* To break the symmetry, set the weights at random.  Each neuron can then learn a different function from its inputs.

* Different initializations produce very different outcomes.Different initializations produce very different outcomes.

* He initialization works well for networks with ReLU activations


| Initialization Method              | Train Accuracy | Test Accuracy | Note                     |
|------------------------------------|----------------|---------------|--------------------------|
| Zeros Initialization               | 50%            | 50%           | Network fails to learn   |
| Large Random Values Initialization | 85%            | 82%           | Weights are too large    |
| He Initialization                  | 95%            | 94%           |Effective for ReLU activation |

