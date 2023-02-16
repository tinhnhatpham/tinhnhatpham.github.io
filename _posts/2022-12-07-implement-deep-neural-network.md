---
layout: post # You can ommit this if you've set it as a default
title: Implement Vectorized Deep Neural Network from scratch
categories: [Deep Learning]
comments: true
---

# Table of contents

- [Install Packages](#install-packages)
- [Introduction](#introduction)
- [Deep Neural Network with multiple Layers](#deep-neural-network-with-multiple-layers)
  - [Forward and Backward Propagation](#forward-and-backward-propagation)
  - [Initialize parameters](#initialize-parameters)
- [Layer Definition](#layer-definition)
  - [Base Layer Class](#base-layer-class)
  - [Linear Layer](#linear-layer)
    - [Forward Propagation](#forward-propagation)
    - [Backward Propagation](#backward-propagation)
  - [Activation Function](#activation-function)
  - [Activation Layer](#activation-layer)
    - [Forward Propagation](#activation-forward-propagation)
    - [Backward Propagation](#activation-backward-propagation)
  - [Dense Layer - Combination of Linear and Activation Layer](#dense-layer)
- [Loss Function](#loss-function)
- [Deep Neural Network Implementation](#deep-neural-network-implementation)
- [Dataset Preparation](#dataset-preparation)
- [Train and Evaluate Model](#train-and-evaluate-model)
- [Conclusion](#conclusion)

## Install Packages <a name="install-packages"></a>


```python
import numpy as np
from scipy.special import expit 
from scipy.special import xlogy
from scipy.special import xlog1py
import matplotlib.pyplot as plt
```

## Introduction  <a name="introduction"></a>


A deep neural network is a type of artificial neural network that is composed of many layers of interconnected nodes. Each layer processes the input data and passes it on to the next layer, until the final layer produces the output.

In this section, we will build from the ground up a deep neural network with multiple hidden layers. The network architecture is depicted in the figure below.

![dnn_architechure](https://docs.google.com/uc?id=18YwZOOKv1Mz8pEt2eJ_HOJKGAp-j3erB)

* The information (input layer) flows through the intermediate layers (hidden layers), and the outcome is determined by the output layer.

* The image above depicts a deep neural network with five layers: one input layer, three hidden layers, and one output layer. We are going to implement a general neural network with one input layer, $n$ hiddens layer, and one output layer.

* Each layer contains its own set of neurons (units). The number of units in the input layer equals the number of sample features. 

Let take a look of one neuron in hidden layer and output layer.

![neuron_architechure](https://docs.google.com/uc?id=1CWPEVaBDUItV4uUxzTWjuxAL8Xd3HfB8)


* The input data ($X$) will go through a linear operation with matrix $W$ (weights) and $b$ (bias):

  $$
    Z = WX + b
  $$

* The result $Z$ is then fed into a nonlinear function known as the activation function (sigmoid, tanh, relu...):

  $$
    A = g(Z)
  $$

* The output $A$ is fed into the next layer until it reaches the output layer.

After we get the output $A$ (prediction) from the output layer, we then compute the different between the actual values and the predicted values.

* Loss function: Used when we refer to the error for a single training example.

* Cost function: Used to refer to an average of the loss functions over an entire training dataset.

To optimize the Cost function and find the model's minimum error value, we will use the gradient descent algorithm. That mean we will update the parameters $W$ and $b$ by their derivative with respect to the Cost.

## Deep Neural Network with multiple Layers <a name="deep-neural-network-with-multiple-layers"></a>



### Forward and Backward Propagation <a name="forward-and-backward-propagation"></a>

Forward propagation and backward propagation are two fundamental concepts in neural networks.

**Forward propagation** refers to the process of calculating the output of a neural network given an input. It involves passing the input through the layers of the network, performing calculations and transformations at each layer, and finally producing an output. The calculations performed during forward propagation are based on the weights and biases of the connections between neurons in the network, which are learned through the training process.

**Backward propagation**, also known as backpropagation, is the process of adjusting the weights and biases of the connections in the network based on the error between the predicted output and the true output. It involves calculating the gradient of the error with respect to the weights and biases, and using this gradient to update the weights and biases in a way that reduces the error. 

The figure below shows the forward and backward propagation in one iteration (epoch). An epoch is a single pass through the entire training dataset. During an epoch, the model processes each example in the training dataset and updates its internal parameters based on the error between the predicted output and the true output. The number of epochs is a hyperparameter that determines how many times the model will pass through the training dataset during the training process.

![dnn_forward_backward_prop](https://docs.google.com/uc?id=1HcFCNsGrI_nfY9aTuV2ssaG-CU8nznvd)

Before we implement forward and backward probagation, let us calculate the shape of parameters $W$ and $b$ in each layer so that we can initialize them later.

Input ($X$) and output ($Y$) layers for $m$ examples:

$$
  X = \overbrace{\begin{bmatrix}
    x^{(1)}_1 & x^{(2)}_1 & \ldots & x^{(m)}_1 \\
    x^{(1)}_2 & x^{(2)}_2 & \ldots & x^{(m)}_2\\
    \vdots & \vdots & \ldots & \vdots\\
    x^{(1)}_n & x^{(2)}_n & \ldots & x^{(m)}_n\\
\end{bmatrix}}^{m\text{ examples}}\\
$$

$$
   Y = \overbrace{\begin{bmatrix}
y^{(1)} & y^{(2)} & \ldots & y^{(m)}
\end{bmatrix}}^{m\text{ examples}}
$$

![dnn_initialize_params](https://docs.google.com/uc?id=15lUIJA_ItWU6KwsbqUteicDAlrgoe7ec)

The figure above is a deep neural network with $1$ example. Let:

* Layer $0$ is the input layer.

* Layer $1$, $2$, and $3$ are hidden layers.

* Layer $4$ is the output layer.

The shape of $W$ and $b$ are defined in following linear equation:

$$
  W*[\text{input}]+b
$$

Let's call $A$ is the input and $Z$ is the linear output of any given layer, we have:

* Layer $1$: $W_{(3,2)}A_{(2,1)}+b_{(3,1)}=Z_{(3,1)}$

* Layer $2$: $W_{(4,3)}A_{(3,1)}+b_{(4,1)}=Z_{(4,1)}$

* Layer $3$: $W_{(3,4)}A_{(4,1)}+b_{(3,1)}=Z_{(3,1)}$

* Layer $4$: $W_{(1,3)}A_{(3,1)}+b_{(1,1)}=Z_{(1,1)}$

**Generalization for $m$ example:**

$$
  W_{\text{(n_units, m)}}\times A_{(\text{n_input}, m)}+b_{(\text{n_units},m)}=Z_{(\text{n_units},m)}\\
$$

Now we have the shapes of $W$ and $b$ in each layer, next step is defined initialize methods for them.

### Initialize parameters <a name="initialize-parameters"></a>

Deep neural networks have two sets of parameters that must be optimized (learned) during the training process. Each layer of the model has two sets of parameters: weights $W$ and bias $b$. Bias $b$ will be initialized to $0$.

The ```Initializer``` class initializes and return $W$ and $b$ parameters by using many difference methods.

**Random normal**

* The weights are initialized with random variable (from standard normal distribution).

* The reason we don't initialize the weights with $0$ is that all the hidden units will become symmetric. No matter how long we update gradient descent, all units are compute the same function.

* The weights also multiply with a small number $(0.01)$ to make its values small and close to $0$. If the weights are too large or too small, which causes tanh or sigmoid activation function to be sarturated. When compute the activations values (tanh, sigmoid, etc) and do a backward propagation, the gradient descent will be very slow.

**Xavier Initialization**

* Xavier method is usually applied to hidden layer with Hyperbolic Tangent activation. $W$ is draw from truncated normal distribution:

  $$
    W_l \sim N\left(0, \frac{1}{\sqrt{n_{l-1}}}\right)\\
    \begin{cases}
      l&\text{layer }l^{th}\\
      n_{l-1}&\text{ number of units in previous layer}
    \end{cases}
  $$

* To achieve that, we can draw $W$ from standard normal distribution $N(0,1)$, and multiply it by $\sqrt{\frac{1}{n_{l-1}}}$.

**He Initialization**

* He initialization is best method for hidden layer with ReLU activation. $W$ is draw from truncated normal distribution:

  $$
    W_l \sim N\left(0, \sqrt{\frac{2}{n_{l-1}}}\right)\\
    \begin{cases}
      l&\text{layer }l^{th}\\
      n_{l-1}&\text{ number of units in previous layer}
    \end{cases}
  $$

* To achieve that, we can draw $W$ from standard normal distribution $N(0,1)$, and multiply it by $\sqrt{\frac{2}{n_{l-1}}}$.


```python
class Initializer():
  def __init__(self, method="random_normal"):
    self.method = method
    self.shape = None

  def __call__(self, shape):
    self.shape = shape
    func_name = self.method
    return getattr(self, func_name)()

  def random_normal(self):
    epsilon = 0.01
    weights = np.random.randn(*self.shape) * epsilon
    bias = np.zeros((self.shape[0], 1)) * epsilon
    return weights, bias

  def xavier(self):
    epsilon = np.sqrt(1/self.shape[1])
    weights = np.random.randn(*self.shape) * epsilon
    bias = np.zeros((self.shape[0], 1)) * epsilon
    return weights, bias
  
  def he(self):
    epsilon = np.sqrt(2/self.shape[1])
    weights = np.random.randn(*self.shape) * epsilon
    bias = np.zeros((self.shape[0], 1)) * epsilon
    return weights, bias
```

## Layer Definition <a name="layer-definition"></a>

### Base Layer Class <a name="base-layer-class"></a>

Let's start with the abstract base layer class, which will define input and output properties, as well as forward and backward methods for all layer classes.



```python
class Layer:
  """Define abtract base layer class."""

  def __init__(self):
    self.input = None
    self.output = None

  # Perform forward propagation for current layer
  def forward_propagation(self, input):
    raise NotImplementedError

  # Perform backward propagation for current layer
  def backward_propagation(self, output, learning_rate=0.01):
    raise NotImplementedError
```

### Linear Layer <a name="linear-layer"></a>



#### Forward Propagation <a name="forward-propagation"></a>

The first thing to do is initialize values for parameters $W$ and $b$ by using ```Initializer``` that we defined above.

![linear_forward](https://docs.google.com/uc?id=1jjb1hxs4kO2zPcpCctE27qKDedUCfksK)


The input of the current layer is the output of the previous layer. The class ```LinearLayer``` computes the the linear transformation in ```forward_propagation``` function as below:

$$
  W*[\text{Input}] + b
$$

#### Backward Propagation <a name="backward-propagation"></a>

![linear_forward](https://docs.google.com/uc?id=1AtOyoPt6eMtVO1WJJEXhc2U7Oc5wSZHo)

Assume that we already have the derivative of output with respect to the cost $\mathcal{L}$. To compute the derivative of $W$ and $b$ w.r.t the cost $\mathcal{L}$, using chain rule in Calculus we have:

$$
  \begin{align}
    \\
    \frac{\partial{\mathcal{L}}}{\partial{W}}&=\frac{\partial{\mathcal{L}}}{\partial_{\text{Output}}}\times\frac{\partial_{\text{Output}}}{\partial{W}}\\
    &=\frac{\partial{\mathcal{L}}}{\partial_{\text{Output}}}\times\text{[Input]}\\
    \\
    \frac{\partial{\mathcal{L}}}{\partial{b}}&=\frac{\partial{\mathcal{L}}}{\partial_{\text{Output}}}\times\frac{\partial_{\text{Output}}}{\partial{b}}\\
    &=\frac{\partial{\mathcal{L}}}{\partial_{\text{Output}}}\\
    \\
  \end{align}
$$

We now have the gradients of the loss ($\frac{\partial{\mathcal{L}}}{\partial_{W}}$ and $\frac{\partial{\mathcal{L}}}{\partial_{b}}$). They are then used to update the parameters to reduce the cost as below:

$$
  \begin{align}
  W &= W - \text{learning_rate}\times\frac{\partial{\mathcal{L}}}{\partial_{W}}\\
  b &= b - \text{learning_rate}\times\frac{\partial{\mathcal{L}}}{\partial_{b}}\\
  \end{align}
$$

Note: ```learning_rate``` is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function.

The last step is to find the derivative of the input w.r.t $\mathcal{L}$.

$$
  \begin{align}
    \frac{\partial{\mathcal{L}}}{\partial_{\text{Input}}}&=\frac{\partial{\mathcal{L}}}{\partial_{\text{Output}}}\times{W}\\
  \end{align}
$$


```python
class LinearLayer(Layer):
  """A layer that performs a linear transformation on its input."""

  def __init__(self, units, initializer='random_normal'):
    self.units = units
    self.initializer = Initializer(initializer)
    self.is_initialized = False

    self.weights = None
    self.bias = None

  def forward_propagation(self, input):
    if not self.is_initialized:
      self.weights, self.bias = self.initializer((self.units, input.shape[0]))
      self.is_initialized = True
      
    self.input = input
    # Linear transformation
    self.output = np.dot(self.weights, input) + self.bias
    return self.output

  def backward_propagation(self, output, learning_rate=0.01):
    m = self.input.shape[1]
    d_weights = 1/m * np.dot(output, self.input.T)
    d_bias = 1/m * np.sum(output, axis=1, keepdims=True)

    # Update parameters
    self.weights -= learning_rate * d_weights
    self.bias -= learning_rate * d_bias

    return np.dot(self.weights.T, output)
```

### Activation Function <a name="activation-function"></a>

An activation function is a mathematical function that is applied to the output of a neuron in a neural network. It is used to determine whether a neuron should be activated or "fired" in response to a given input. Activation functions are a key component of neural networks and are used to introduce nonlinearity into the network.

Let's define some activation functions and their derivative for the network. We won't go to detail about how to find the derivative of the activation function. 

1. Sigmoid function:

  The sigmoid function maps any input value to a value between 0 and 1. It is often used in the output layer of a binary classification model, where values close to 0 correspond to one class and values close to 1 correspond to the other class.

  $$
    \sigma(x) = \frac{1}{1 + e^{-x}}\\
  $$

  <u>Derivative:</u>

  $$
    \sigma'(x) = \sigma(x)\big(1-\sigma(x)\big) \\
  $$

2. ReLU (Rectified Linear Unit):

   The ReLU function outputs the input value if it is positive, and 0 if it is negative. It is a simple and widely used activation function that has been shown to be effective in many deep learning applications.

  $$
    \text{ReLU}(x) = \text{max}(0, x)\\
  $$

  <u>Derivative:</u>

  $$
    f(x)=\begin{cases}
      0&&\text{if }x\gt0\\
      x&&\text{if }x\lt0\\
      \text{undefined}&&x=0\\
    \end{cases}
  $$

3. Tanh (Hyperbolic Tangent) function:

  The tanh function maps any input value to a value between -1 and 1. It is similar to the sigmoid function, but is centered around 0 and has a broader range of output values.

  $$
    \text{tanh}(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}\\
  $$

  <u>Derivative:</u>

  $$
    \text{tanh}'(x) = 1-\text{tanh}^2(x)\\
  $$


```python
class Activation:
  """Define activation functions and their derivative for deep neural network."""

  def __init__(self, name=None):
    self.name = name

  def __call__(self, x, derivative=False):
    if self.name is None:
      return None
    func_name = self.name if not derivative else self.name + "_prime"
    return getattr(self, func_name, None)(x)

  def sigmoid(self, x):
    return expit(x)
  
  def sigmoid_prime(self, x):
    return self.sigmoid(x)*(1 - self.sigmoid(x))

  def relu(self, x):
    return np.maximum(0, x)

  def relu_prime(self, x):
    return 1. * (x > 0)

  def tanh(self, x):
    return np.tanh(x);

  def tanh_prime(self, x):
    return 1-np.tanh(x)**2;    
```

### Activation Layer <a name="activation-layer"></a>

#### Forward Propagation <a name="activation-forward-propagation"></a>

![activation_forward](https://docs.google.com/uc?id=117672JG67Q9H_y9z4NYEpbnD20mZSp-_)

In activation layer, forward propagation is simple pass the input through activation function ($g$).

#### Backward Propagation <a name="activation-backward-propagation"></a>

![activation_backward](https://docs.google.com/uc?id=1edw7eEueAxUhe6gFkskhlGiQdXCEVp35)

The activation layer doesn't have any parameter to learn, the input pass through an activation function which is non-linearity. In order to perform a backward propagation, we take the derivative of the activation function w.r.t Input.

$$
  \begin{align}
    \frac{\partial{\mathcal{L}}}{\partial_\text{Input}}&=\frac{\partial{\mathcal{L}}}{\partial_\text{Output}}\times\frac{\partial_\text{Output}}{\partial_\text{Input}}\\
  \end{align}
$$


```python
class ActivationLayer(Layer):
  """A layer that applies a non-linear transformation to the input."""
  
  def __init__(self, activation_name=None):
    self.activation = Activation(activation_name)

  def forward_propagation(self, input):
    self.input = input
    self.output = self.activation(input)
    return self.output

  def backward_propagation(self, output):
    return output * self.activation(self.input, derivative=True)
```

### Dense Layer - Combination of Linear and Activation Layer <a name="dense-layer"></a>

After we've completed the Linear and Activation Layers, we'll combine them to form the Dense Layer, as shown below:

![full_connected_layer](https://docs.google.com/uc?id=1YsQ56vNycJv7Hn1pt_6LZ53up8U855xA)


```python
class Dense(Layer):
  """A layer that combines linear and non-linear transformation of its input."""
  
  def __init__(self, units, activation, initializer='random_normal'):
    self.linearLayer = LinearLayer(units, initializer)
    self.activationLayer = ActivationLayer(activation)
    
  def forward_propagation(self, input):
    self.input = input
    self.output = self.linearLayer.forward_propagation(input)
    self.output = self.activationLayer.forward_propagation(self.output)
    return self.output

  def backward_propagation(self, output, learning_rate=0.01):
    self.output = self.activationLayer.backward_propagation(output)
    self.output = self.linearLayer.backward_propagation(self.output, learning_rate)
    return self.output
```

## Loss Function <a name="loss-function"></a>

We now have forward and backward algorithms; the only thing missing is the Loss Function. For our classification, we'll use the the cross-entropy loss.

Cross-entropy loss, also known as log loss, is a common loss function used in classification tasks. It measures the difference between the predicted probability distribution and the true probability distribution. The true probability distribution is represented by the one-hot encoded labels, while the predicted probability distribution is represented by the predicted probabilities of the input data belonging to each class.

The formula for cross-entropy loss is given by:

$$
    \mathcal{L}\left(\widehat{y},y\right)=-\left(y\log\widehat{y}+(1-y)\log(1-\widehat{y})\right)\\
$$

We want the cost to be optimized as small as possible so that $y$ and $\widehat{y}$ close together.

$$
    \\
    \begin{cases}
        &y=1&-\mathcal{L}\left(\widehat{y},y\right)=-\log\widehat{y}&\leftarrow\widehat{y}\text{ need to be large}\\
        &y=0&-\mathcal{L}\left(\widehat{y},y\right)=-\log(1-\widehat{y})&\leftarrow\widehat{y}\text{ need to be small}\\
    \end{cases}
    \\
$$

The cost is the computed by averaging all costs of entire set:

$$
    \begin{align}\\
        \mathcal{L}(\widehat{y},y)&=\frac{1}{m}\sum_{i=1}^m\mathcal{L}\left(\widehat{y}^{(i)},y^{(i)}\right)\\
        &=-\frac{1}{m}\sum_{i=1}^my^{(i)}\log\widehat{y}^{(i)}+(1-y^{(i)})\log(1-\widehat{y}^{(i)})\\
    \end{align}
$$

The derivative of $\mathcal{L}$ w.r.t $\widehat{y}$ is:

$$
    \begin{align}
        \frac{\partial{\mathcal{L}}}{\partial{\widehat{y}}}&=-y\frac{1}{\widehat{y}}-(1-y)\frac{1}{1-\widehat{y}}-1\\
        &=\frac{-y}{\widehat{y}}+\frac{1-y}{1-\widehat{y}}\\
    \end{align}
$$


```python
class Loss():
  """Compute the cost of the predicted output of the model and the true output."""
  
  def __init__(self, name):
    self.name = name
  
  def __call__(self, Y, Y_pred, derivative=False):
    func_name = self.name if not derivative else self.name + "_prime"
    return getattr(self, func_name)(Y, Y_pred)

  def crossentropy(self, Y, Y_pred):
    m = Y.shape[1]
    # Add a very small epsilon to avoid log(0)
    epsilon = 1e-5
    cost = np.multiply(Y, np.log(Y_pred + epsilon)) + np.multiply(1 - Y, np.log(1 - Y_pred + epsilon))
    return np.squeeze(-1/m * np.sum(cost))

  def crossentropy_prime(self, Y, Y_pred):
    epsilon = 1e-5
    return - (np.divide(Y, Y_pred + epsilon) - np.divide(1 - Y, 1 - Y_pred + epsilon))
```

## Deep Neural Network Implementation <a name="deep-neural-network-implementation"></a>

We have everything we need so far. Let put every we have till now to implement a Deep Neural Network.


```python
class NeuralNetwork():
  """Deep neural network with multiple layers."""

  def __init__(self):
    self.layers = []
    self.loss = None
    self.costs = []
    self.learning_rate = None

  def add(self, layer):
    """Add layer to the network.

    Args: 
      layer: A layer to be added to the network
    """
    self.layers.append(layer)

  # Select the loss for model before training
  def compile(self, loss):
    """Configurations for training the network.
    
    Args:
      loss: Loss function
    """
    self.loss = Loss(loss)

  def fit(self, X, Y, epochs=1000, learning_rate=0.01, print_log=False):
    """Training the network.

    Args:
      X: Training example 
      Y: Ground truth label
      epochs: Number of loop through entire X
      learning_rate: Step size for updating parameters
      print_log: Print training log
    """
    self.learning_rate = learning_rate
    cost = 0

    for i in range(epochs):
      # Forward propagation
      output = X
      for layer in self.layers:
        output = layer.forward_propagation(output)
      cost = self.loss(Y, output)

      # Backward propagation
      error = self.loss(Y, output, derivative=True)
      for layer in reversed(self.layers):
        error = layer.backward_propagation(error, learning_rate)
      
      if (i%100 == 0 or i == epochs - 1):
        self.costs.append(cost)

        if print_log: 
          print(f"Epochs: {i} - Cost: {cost}")
         
  def predict(self, X):
    """Predict output with given input."""
    input = X
    for layer in self.layers:
      input = layer.forward_propagation(input)
    return input

  def evaluate(self, X, Y):
    """Evaluation the network."""
    result = {"accuracy": (100 - np.mean(np.abs(Y - self.predict(X)))*100)}
    return result
```

## Dataset Preparation <a name="dataset-preparation"></a>

To test our neural network, we will use the <i>Cat vs Non-Cat</i> dataset, you can download the dataset from [here](https://www.kaggle.com/datasets/mriganksingh/cat-images-dataset). 


```python
import h5py

def load_data():
    train_dataset = h5py.File('/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
```

Let’s take a look at our training set to get a better sense of what cat and non-cat dataset look like. We will display 8 images for each type.


```python
plt.figure(figsize=(12,12)) # specifying the overall grid size

for i in range(16):
    plt.subplot(4,4,i+1)  
    plt.imshow(train_x_orig[i])
    label = "Cat" if train_y[0][i] == 1 else "Non-cat"
    plt.title(label)
    plt.axis('off')
plt.show()
```


    
![png](\assets\images\blogs\2022-12-07-implement-deep-neural-network-output_29_0.png)
    


Each image in the training and test sets has three channels (red, green, and blue). We flatten it so that each pixel is a feature. Each flattened image was also vertically stacked, as shown in the figure above.

![full_connected_layer](https://docs.google.com/uc?id=1YYKlIa69e9l2cIogMPx8GhoYYRIu1ASL)



```python
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
```

We standardize the datasets before feeding to the neural network (model). For image datasets, all the images are in range [0,255], so we only need to divide the images to 255 (the maximum value of a pixel).


```python
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
```

    train_x's shape: (12288, 209)
    test_x's shape: (12288, 50)


## Train and Evaluate Model <a name="train-and-evaluate-model"></a>


```python
# Create model
dnn = NeuralNetwork()

# Add layers
dnn.add(Dense(units=20, activation="relu", initializer="he"))
dnn.add(Dense(units=7, activation="relu", initializer="he"))
dnn.add(Dense(units=5, activation="relu", initializer="he"))
dnn.add(Dense(units=1, activation="sigmoid"))

# Choose loss function
dnn.compile(
    loss = "crossentropy"
)

# Train model
dnn.fit(train_x, train_y, epochs=2500, learning_rate=0.005, print_log=True)
```

    Epochs: 0 - Cost: 0.6934400094242016
    Epochs: 100 - Cost: 0.677607890477213
    Epochs: 200 - Cost: 0.645620195779834
    Epochs: 300 - Cost: 0.6350100406529784
    Epochs: 400 - Cost: 0.6115859808114084
    Epochs: 500 - Cost: 0.56829023451606
    Epochs: 600 - Cost: 0.521586003304234
    Epochs: 700 - Cost: 0.47601939324370324
    Epochs: 800 - Cost: 0.43561744470339575
    Epochs: 900 - Cost: 0.4210744831897455
    Epochs: 1000 - Cost: 0.3788573130182903
    Epochs: 1100 - Cost: 0.34617996290825126
    Epochs: 1200 - Cost: 0.3098894757643255
    Epochs: 1300 - Cost: 0.26936597663148487
    Epochs: 1400 - Cost: 0.2470517680508833
    Epochs: 1500 - Cost: 0.23381347851204337
    Epochs: 1600 - Cost: 0.23095143234415358
    Epochs: 1700 - Cost: 0.19582886134312555
    Epochs: 1800 - Cost: 0.18629917601481902
    Epochs: 1900 - Cost: 0.17916329257451358
    Epochs: 2000 - Cost: 0.16812191554455946
    Epochs: 2100 - Cost: 0.16181214997307128
    Epochs: 2200 - Cost: 0.15088690768523802
    Epochs: 2300 - Cost: 0.14737561282563494
    Epochs: 2400 - Cost: 0.13820776441591212
    Epochs: 2499 - Cost: 0.14001890492622304



```python
print("Train", dnn.evaluate(train_x, train_y))
print("Test", dnn.evaluate(test_x, test_y))
```

    Train {'accuracy': 88.69425346212275}
    Test {'accuracy': 52.60689936119567}


Let plot the learning curve with the cost function and the gradients.


```python
plt.plot(dnn.costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 100 hundreds)')
plt.title("Learning rate =" + str(dnn.learning_rate))
plt.show()
```


    
![png](\assets\images\blogs\2022-12-07-implement-deep-neural-network-output_38_0.png)
    


## Conclusion <a name="conclusion"></a>

The cost is declined during the training, that means our model can learned through gradient descent algorithm. 

Our model performs well on the train set, with an accuracy of 88.69%, but poorly on the test set, with only 52.60%. Our model can be trained further to improve, but we can see that it suffered from overfitting.

Next time, we'll improve and optimize our neural network to combat overfitting and enable faster training.