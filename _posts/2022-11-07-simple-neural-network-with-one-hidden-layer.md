---
layout: post # You can ommit this if you've set it as a default
title: Implement Simple Neural Network with One Hidden Layer
categories: [Deep Learning]
comments: true
---


## 1. Install Packages


```python
# Install Jupyter Dynamic Classes package
!pip install jdc
```


```python
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt
import numpy as np
import jdc

%matplotlib inline
```


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

## 2. Data Processing

For the purpose of this blog, we use ```make_moons``` function to generate a binary dataset.

Reshape the training dataset $X$ and ground truth $Y$ so they have the shape:

$$
  X = \overbrace{\begin{bmatrix}
x^{(1)}_1 & x^{(2)}_1 & \ldots & x^{(n)}_1 \\
x^{(1)}_2 & x^{(2)}_2 & \ldots & x^{(n)}_2
\end{bmatrix}}^{\text{m examples}}\\
$$

$$
   Y = \overbrace{\begin{bmatrix}
y^{(1)} & y^{(2)} & \ldots & y^{(m)}
\end{bmatrix}}^{\text{m examples}}
$$


```python
# Generate 2d classification dataset
X, Y = sklearn.datasets.make_moons(n_samples=400, noise=0.2)

X = X.T
Y = np.reshape(Y, (1, -1))

print("X shape: ", X.shape)
print("Y shape: ", Y.shape)
print("m examples: ", X.shape[1])
```

    X shape:  (2, 400)
    Y shape:  (1, 400)
    m examples:  400


Let's plot the dataset.


```python
# scatter plot, dots colored by class value
plt.scatter(X[0, :], X[1, :], c=Y, s=15, cmap=plt.cm.Spectral);
```


    
![png](\assets\images\blogs\2022-11-07-shallow-neural-network-with-one-hidden-layer-1.png)
    


## 3. Logistic Regression

Let's try the Logistic Regression model on our data before building our Neural Netword model to see how its classification works on the datasets.  

We're going to use the Logistic Regression built-in function from sklearn.



```python
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T.ravel())

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
```


    
![png](\assets\images\blogs\2022-11-07-shallow-neural-network-with-one-hidden-layer-2.png)
    



```python
# Print accuracy of the model
print("Accuracy of Logistic Regression Model: ", clf.score(X.T, Y.T)*100, "%.")
```

    Accuracy of Logistic Regression Model:  87.0 %.


The Logistic Regression model fails to capture the moon shape because the dataset is not linearly separable. Let's take a look at how a neural network with only one hidden layer performs.

## 4. Neural Network with One Hidden Layer

Let's implement a neural network with one hidden layer, and see how it performs on our non-linear dataset.


![png](\assets\images\blogs\2022-11-07-shallow-neural-network-with-one-hidden-layer-3.png)
<caption><center><font color='gray'><b>Figure 1:</b>The model architechure</font></center></caption>

**Notation:**

$$
  \begin{align}
    &x^i&&\text{$i^{th}$ training example}\\
    &w^{[n](i)}&&\text{weight of $n^{th}$ layer for $i^{th}$ training example}\\
    &b^{[n](i)}&&\text{bias of $n^{th}$ layer for $i^{th}$ training example}\\
    &z^{[n](i)}&&\text{$wa+b$ of $n^{th}$ layer for $i^{th}$ training example}\\
    &\widehat{y}^{(i)}&&\text{model prediction for $i^{th}$ training example}
  \end{align}\\
$$

**Mathematical Expression for one example $x^i$:**

$$
  \begin{align}
    z^{[1](i)}&=W^{[1]}x{(i)}+b^{(i)}\\
    a^{[1](i)}&=\text{tanh}\left(z^{[1](i)}\right)\\
    z^{[2](i)}&=W^{[2]}a^{[1](i)}+b^{[2]}\\
    \widehat{y}^{(i)}&=a^{[2](i)}=\sigma\left(z^{[2](i)}\right)\\
    y^{(i)}_{\text{predictions}}&=\begin{cases}
    1 & \text{if }a^{[2](i)}\gt0.5\\
    0 & \text{otherwise}\\
    \end{cases}
  \end{align}\\
$$

Apply Cross-entropy loss (log loss) for the entire example predictions to compute the lost $J$:

$$\\
  J=-\frac{1}{m}\sum_{i=1}^{m}\left(y^{(i)}\log\left(a^{[2](i)}\right)+(1-y^{(i)})\log\left(1-a^{[2](i)}\right)\right)\\
$$

**Building neural network steps**

* Define model architechure (input shape, hidden units, etc).

* Initialize parameters (weights, bias...).

* Loop: 

    * Forward propagation.

    * Compute loss.

    * Backward propagation (get gradients).

    * Update parameters.

    * Training model.

### 4.1 Define mode architechure

Our model includes three layers:

* Input layer (number of features).

* Hidden layer (number of hidden nodes).

* Output layer (number of predictions: 0 - 1)


```python
class OneHiddenLayerNN():
  """
  A simple neural network with one hidden layer.
  """
  def __init__(self, n_input_size, n_hidden_size, n_output_size):
    """
    Define model layers and initialize parameters.
    
    Arguments:
    n_input_size -- number of input (features)
    n_hidden_size -- number of nodes in the hidden layer
    n_output_size -- number of output layer
    """

    # Define model layers
    self.n_input_size      = n_input_size
    self.n_hidden_size     = n_hidden_size
    self.n_output_size     = n_output_size
    
    self.Z1 = None
    self.A1 = None
    self.Z2 = None
    self.A2 = None

    self.dW1 = None
    self.db1 = None
    self.dW2 = None
    self.db2 = None

    self.cost = None
    self.learning_rate = None

    # Initialize parameters
    self.parameters_initialize()

  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
```

### 4.2 Parameters Initialization

Clarify the shapes for $\displaystyle W^{[1]}$ and $\displaystyle b^{[1]}$ with 1 examples $(m=1)$

$$
\underbrace{\begin{bmatrix}
w_{1,1} & w_{1,2} \\
w_{2,1} & w_{2,2} \\
w_{3,1} & w_{3,2} \\
w_{4,1} & w_{4,2} \\
\end{bmatrix}}_{(\text{n_h, n_x})}\times
\underbrace{\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}}_{\text{(n_x, m)}}+
\underbrace{\begin{bmatrix}
b_{1} \\
b_{2}  \\
b_{3}  \\
b_{4} 
\end{bmatrix}}_{\text{(n_h, 1)}}=
\underbrace{\begin{bmatrix}
(w_{1,1}x_1 +  w_{1,2}x_2) + b_1\\
(w_{2,1}x_1 +  w_{2,2}x_2) + b_2 \\
(w_{3,1}x_1 +  w_{3,2}x_2) + b_3 \\
(w_{4,1}x_1 +  w_{4,2}x_2) + b_4
\end{bmatrix}}_{\text{(n_h, m)}}
$$

Clarify the shapes for $\displaystyle W^{[2]}$ and $\displaystyle b^{[2]}$ with 1 examples $(m=1)$

$$
\underbrace{\begin{bmatrix}
w_1 & w_2 & w_3 & w_4\\
\end{bmatrix}}_{(\text{1, n_h})}\times
\underbrace{\begin{bmatrix}
a^{[2]_1}\\
a^{[2]_2}\\
a^{[2]_3}\\
a^{[2]_4}\\
\end{bmatrix}}_{\text{(n_h, m)}}+
\underbrace{\begin{bmatrix}
b_{2} \\
\end{bmatrix}}_{\text{(1, m)}}=\\   
\underbrace{\begin{bmatrix}
(w_{1} + a^{[2]_1})+b_1&(w_{2} + a^{[2]_2})+b_2&(w_{3} + a^{[2]_3})+b_3&(w_{4} + a^{[2]_4})+b_4\\
\end{bmatrix}}_{\text{(n_y, m)}}
$$

* The weights are initialized with random variable (from standard normal distribution).

* The reason we don't initialize the weights with $0$ is that all the hidden units will become symmetric. No matter how long we update gradient descent, all units are compute the same function.

* The weights also multiply with a small number $(0.01)$ to make its values small and close to $0$. If the weights are too large or too small, which causes tanh or sigmoid activation function to be sarturated. When compute the activations values (tanh, sigmoid, etc) and do a backward propagation, the gradient descent will be very slow.


```python
%%add_to OneHiddenLayerNN
def parameters_initialize(self):
  self.W1 = np.random.randn(self.n_hidden_size, self.n_input_size) * 0.01
  self.b1 = np.zeros((self.n_hidden_size, 1))
  self.W2 = np.random.randn(self.n_output_size, self.n_hidden_size) * 0.01
  self.b2 = np.zeros((self.n_output_size, 1))
```

### 4.3 The Loop

#### 4.3.1 Implement Forward Propagation

**Mathematical expression for entire training set $X$**

$$
  \text{Shapes}\begin{cases}
    X&\text{(n_x, m)}\\
    Y&\text{(n_y, m)}\\
    W^{[1]}&\text{(n_h, n_x)}\\
    b^{[1]}&\text{(n_h, 1)}\\
    Z^{[1]}&\text{(n_h, m)}\\
    A^{[1]}&\text{(n_h, m)}\\
    W^{[2]}&\text{(n_y, n_h)}\\
    b^{[2]}&\text{(n_y, 1)}\\
    Z^{[2]}&\text{(n_y, m)}\\
    A^{[2]}&\text{(n_y, m)}\\
  \end{cases}
$$

$$
  \begin{align}
    \\
    &\qquad{X}&&\text{(n_x, m)}\\
    &\qquad\huge{⇓}\\
    Z^{[1]}&=W^{[1]}X + b^{[1]}&&\text{(n_h, m)}\\
    &\qquad\huge{⇓}\\
    A^{[1]}&=\text{tanh}\left(Z^{[1]}\right)&&\text{(n_h, m)}\\
    &\qquad\huge{⇓}\\
    Z^{[2]}&=W^{[2]}A^{[1]} + b^{[2]}&&\text{(n_h, m)}\\
    &\qquad\huge{⇓}\\
    A^{[2]}&= \widehat{Y}=\sigma\left(Z^{[2]}\right)&&\text{(n_h, m)}\\
  \end{align}
$$


```python
%%add_to OneHiddenLayerNN
def forward_propagation(self, X):
  """
  Compute forward propagation

  Argument:
  X -- input data of size (self.n_input_size, m)
  """
  self.Z1 = np.dot(self.W1, X) + self.b1
  self.A1 = np.tanh(self.Z1)
  self.Z2 = np.dot(self.W2, self.A1) + self.b2
  self.A2 = self.sigmoid(self.Z2)
```

#### 4.3.2 Compute the Cost

The ```forward_propagation``` function has computed $\widehat{Y}$ $(A^{[2]})$. Now we can compute the loss $J$ across entire training set by average the loss of each training example.

$$
  \color{darkred}{J=\mathcal{L}(a^{[2]}, y)=-\frac{1}{m}\sum_{i=1}^{m}\left(y^{(i)}\log\left(a^{[2](i)}\right)+(1-y^{(i)})\log\left(1-a^{[2](i)}\right)\right)}\\
$$


```python
%%add_to OneHiddenLayerNN
def compute_cost(self, Y):
  """
  Compute the cost using Cross-Entropy loss

  Argument:
  Y -- true label vector (1 if cat, 0 if dog) with dim (1, number of example)
  """

  m = Y.shape[1]

  logprops = np.multiply(Y, np.log(self.A2)) + np.multiply((1 - Y), np.log(1 - self.A2))

  self.cost = np.squeeze(-1/m * np.sum(logprops))
```

#### 4.3.3 Implement Backward Propagation

We have computed the cost $J$, now we're going to minimize it using Backward Propagation. We'll do so with gradient descent:

* Start with parameters that has been initialized with random values.

* Evaluate the gradient of the loss $J$ with respected to the paramenters ($\frac{\partial{\mathcal{L}}}{\partial{W}}$ and $\frac{\partial{\mathcal{L}}}{\partial{b}}$).

* Update the parameters to decrease the loss $J$.

![png](\assets\images\blogs\2022-11-07-shallow-neural-network-with-one-hidden-layer-4.png)
<caption><center><font color='gray'><b>Figure 2:</b> Backward Propagation for 1 example</font></center></caption>

In figure 2, by the Chain Rule in Calculus, we can calculate gradient descent for 1 example as following:

$$
  \begin{align}
    \frac{\partial{\mathcal{L}}}{\partial{W^{[2]}}}&=\color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{a^{[2]}}}\times\frac{\partial{a^{[2]}}}{\partial{z^{[2]}}}\times\frac{\partial{z^{[2]}}}{\partial{W^{[2]}}}}\\
    \frac{\partial{\mathcal{L}}}{\partial{b^{[2]}}}&=\color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{a^{[2]}}}\times\frac{\partial{a^{[2]}}}{\partial{z^{[2]}}}\times\frac{\partial{z^{[2]}}}{\partial{b^{[2]}}}}\\
    \\
    \frac{\partial{\mathcal{L}}}{\partial{W^{[1]}}}&=\color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{a^{[2]}}}\times\frac{\partial{a^{[2]}}}{\partial{z^{[2]}}}\times\frac{\partial{z^{[2]}}}{\partial{a^{[1]}}}\times\frac{\partial{a^{[1]}}}{\partial{z^{[1]}}}\times\frac{\partial{z^{[1]}}}{\partial{W^{[1]}}}}\\
    \frac{\partial{\mathcal{L}}}{\partial{b^{[1]}}}&=\color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{a^{[2]}}}\times\frac{\partial{a^{[2]}}}{\partial{z^{[2]}}}\times\frac{\partial{z^{[2]}}}{\partial{a^{[1]}}}\times\frac{\partial{a^{[1]}}}{\partial{z^{[1]}}}\times\frac{\partial{z^{[1]}}}{\partial{b^{[1]}}}}\\\\
  \end{align}
$$

Our neural netword using $\text{tanh}$ for activation and $\text{sigmoid}$ for classification, their deravations are:

$\displaystyle
  \begin{align}
  \sigma'(x)=\sigma(x)(1-\sigma(x))\\
  \text{tanh}'(x)=1-\text{tanh}^2(x)\\
  \end{align}
$


Let's calculate partial derivatives:

$\qquad\displaystyle
  \color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{a^{[2]}}}=\frac{a^{[2]}-y}{a^{[2]}(1-a^{[2]})}}\qquad(1)\\
$

<details>
  <summary><font color='#ADECEE'>Click for solution</font></summary>

$$
    \begin{align}
        \frac{\partial{\mathcal{L}}}{\partial{a^{[2]}}}&=-y\frac{1}{a^{[2]}}-(1-y)\frac{1}{1-a^{[2]}}-1\\
        &=\frac{-y}{a^{[2]}}+\frac{1-y}{1-a^{[2]}}\\
        &=\frac{-y(1-a^{[2]})+a^{[2]}(1-y)}{a^{[2]}(1-a^{[2]})}\\
        &=\frac{a^{[2]}-y}{a^{[2]}(1-a^{[2]})}
    \end{align}
$$

</details>


$\qquad\displaystyle
  \color{darkred}{\frac{\partial{a^{[2]}}}{\partial{z^{[2]}}}=a^{[2]}(1-a^{[2]})}\qquad(2)\\
$

<details>
  <summary><font color='#ADECEE'>Click for solution</font></summary>

$$
  \begin{align}
  \sigma'(x)&=\sigma(x)(1-\sigma(x))\\
  a^{[2]} &= \sigma(z^{[2]})\\
  \implies \frac{\partial{a^{[2]}}}{\partial{z^{[2]}}}&=a^{[2]}(1-a^{[2]})
  \end{align}
$$

</details>

$\qquad\displaystyle
  \color{darkred}{\frac{\partial{z^{[2]}}}{\partial{a^{[1]}}}=W^{[2]}}\qquad(3)\\
$

$\qquad\displaystyle
  \color{darkred}{\frac{\partial{a^{[1]}}}{\partial{z^{[1]}}}=1-(a^{[1]})^2}\qquad(4)\\
$

<details><summary><font color='#ADECEE'>Click for solution</font></summary>

$$
  \begin{align}
    \text{tanh}'(x)&=1-\text{tanh}^2(x)\\
    a^{[1]}&=\text{tanh}(z^{[1]})\\
    \implies \frac{\partial{a^{[1]}}}{\partial{z^{[1]}}}&=1-(a^{[1]})^2\\
  \end{align}
$$

</details>

$\qquad\displaystyle
    \color{darkred}{\frac{\partial{z^{[2]}}}{\partial{W^{[2]}}}}\color{darkred}{=a^{[1]}}\qquad(5)\\
$

$\qquad\displaystyle\color{darkred}{\frac{\partial{z^{[2]}}}{\partial{b^{[2]}}}}\color{darkred}{=1}\qquad(6)\\
$

$\qquad\displaystyle\color{darkred}{\frac{\partial{z^{[1]}}}{\partial{W^{[1]}}}}\color{darkred}{=x}\qquad(7)\\
$

$\qquad\displaystyle\color{darkred}{\frac{\partial{z^{[1]}}}{\partial{b^{[1]}}}}\color{darkred}{=1}\qquad(8)\\
$



Use Chain Rule to calculate the derivative of the loss J with respected to $W$ and $b$:

$$
  \begin{align}
  \frac{\partial{\mathcal{L}}}{\partial{W^{[2]}}}&=(1)\times(2)\times(5)&&=(a^{[2]}-y){a^{[1]}}^T\\
  \frac{\partial{\mathcal{L}}}{\partial{b^{[2]}}}&=(1)\times(2)\times(6)&&=a^{[2]}-y\\
  \frac{\partial{\mathcal{L}}}{\partial{W^{[1]}}}&=(1)\times(2)\times(3)\times(4)\times(7)&&={W^{[2]}}^T\left(a^{[2]}-y\right)*(1-{a^{[1]}}^2)x^T\\
  \frac{\partial{\mathcal{L}}}{\partial{b^{[1]}}}&=(1)\times(2)\times(3)\times(4)\times(7)&&={W^{[2]}}^T\left(a^{[2]}-y\right)*(1-{a^{[1]}}^2)\\
  \end{align}
$$

Let's calculate gradient descent for $m$ example:

$$
    \text{Shapes}\begin{cases}
      X&\text{(n_x, m)}\\
      Y&\text{(n_y, m)}\\
      W^{[1]}&\text{(n_h, n_x)}\\
      b^{[1]}&\text{(n_h, 1)}\\
      Z^{[1]}&\text{(n_h, m)}\\
      A^{[1]}&\text{(n_h, m)}\\
      W^{[2]}&\text{(n_y, n_h)}\\
      b^{[2]}&\text{(n_y, 1)}\\
      Z^{[2]}&\text{(n_y, m)}\\
      A^{[2]}&\text{(n_y, m)}\\
  \end{cases}
  \\
  \begin{align}
  \\
  \color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{W^{[2]}}}}&\color{darkred}{=(A^{[2]}-Y){A^{[1]}}^T}&&\text{(n_y, n_h)}\\
  \color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{b^{[2]}}}}&\color{darkred}{=A^{[2]}-Y}&&\text{(n_y, m)}\\
  \color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{W^{[1]}}}}&\color{darkred}{={W^{[2]}}^T\left(A^{[2]}-Y\right)*(1-{A^{[1]}}^2)X^T}&&\text{(n_h, n_x)}\\
  \color{darkred}{\frac{\partial{\mathcal{L}}}{\partial{b^{[1]}}}}&\color{darkred}{={W^{[2]}}^T\left(A^{[2]}-Y\right)*(1-{A^{[1]}}^2)}&&\text{(n_h, m)}\\
  \end{align}\\
$$


* $\displaystyle \frac{\partial{\mathcal{L}}}{\partial{b^{[1]}}}$ has the shape $\text{(n_h, m)}$, so we will sum across the $y\text{-axis}$ to match with $b_1$ shape $\text{(n_h, 1)}$.

* $\displaystyle \frac{\partial{\mathcal{L}}}{\partial{b^{[2]}}}$ has the shape $\text{(n_y, m)}$, so we will sum across the $y\text{-axis}$ to match with $b_2$ shape $\text{(n_y, 1)}$.

* $J$ is computed by average on entire set (divide by $m$), so we will divide by $m$ for both $\displaystyle \frac{\partial{\mathcal{L}}}{\partial{b^{[1]}}}$ and $\displaystyle \frac{\partial{\mathcal{L}}}{\partial{b^{[2]}}}$.


```python
%%add_to OneHiddenLayerNN
def backward_propagation(self, X, Y):
  """
  Perform backward propagation and compute partial derivatives dW and db
  """

  m = X.shape[1]
  
  dZ2 = self.A2 - Y
  self.dW2 = 1/m * np.dot(dZ2, self.A1.T)
  self.db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

  dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(self.A1, 2))
  self.dW1 = 1/m * np.dot(dZ1, X.T)
  self.db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
```

#### 4.3.3 Update Parameters

The ```backward_propagation``` function has computed the gradient descent for $dW1, db1, dW2, db2$. W update our parameters by following rule:

$$
  \begin{align}
    W^{[1]} &:= W^{[1]} - \alpha\frac{\partial{\mathcal{L}}}{\partial{W^{[1]}}}\\
    W^{[2]} &:= W^{[2]} - \alpha\frac{\partial{\mathcal{L}}}{\partial{W^{[2]}}}\\
    b^{[1]} &:= b^{[1]} - \alpha\frac{\partial{\mathcal{L}}}{\partial{b^{[1]}}}\\
    b^{[2]} &:= b^{[2]} - \alpha\frac{\partial{\mathcal{L}}}{\partial{b^{[2]}}}\\
  \end{align}
$$

* The learning rate $\displaystyle\alpha$ is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function. The figure below demonstrates how good and bad learning rates impact on trainng the model.

![gif](\assets\images\blogs\2022-11-07-shallow-neural-network-with-one-hidden-layer-5.gif)
<caption><center><font color='gray'><b>Figure 3:</b>The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.</font></center></caption>




```python
%%add_to OneHiddenLayerNN
def update_paramenters(self):
  """
  Update the parameters using gradient descent and learning rate
  """

  self.W1 = self.W1 - self.learning_rate * self.dW1
  self.b1 = self.b1 - self.learning_rate * self.db1
  self.W2 = self.W2 - self.learning_rate * self.dW2
  self.b2 = self.b2 - self.learning_rate * self.db2
```

### 4.4 Training model

By now, we have defined all the parts to build a neural network with 1 hidden layer. It's time to put all together.


```python
%%add_to OneHiddenLayerNN
def train(self, X, Y, n_iterations=10000, learning_rate=0.001, print_log=False):
  """

  Arguments:
  X -- input data of size (self.n_input_size, m)
  Y -- true label vector with dim (1, number of example)
  n_iterations -- number of iterations through the data examples
  learning_rate -- step size for each iteration
   print_log -- print the loss every n steps
  """

  self.learning_rate = learning_rate

  for i in range(n_iterations):
    self.forward_propagation(X)
    self.compute_cost(Y)
    self.backward_propagation(X, Y)
    self.update_paramenters()

    if (print_log and i%1000 == 0):
      print("Iteration: %i - Cost: %f" %(i, self.cost))
```

### 4.5 Model Prediction and Evaluation.

#### 4.5.1 Prediction

The ```predict``` function will output predictions for an input set by using the forward propagation with learned parameters that the model has optimized.

Our prediction is binary, which means on $0$ or $1$, so we use some **threshold** to classify them.

$$
  \begin{cases}
    1&\text{if $\widehat{Y}$ > threshold}\\
    0&\text{otherwise}
  \end{cases}
$$


```python
%%add_to OneHiddenLayerNN
def predict(self, X):
  """
  Predict a class for each example in X using learned parameters

  Arguments:
  X -- input data of size (self.n_input_size, m)
  """

  Z1 = np.dot(self.W1, X) + self.b1
  A1 = np.tanh(Z1)
  Z2 = np.dot(self.W2, A1) + self.b2
  A2 = self.sigmoid(Z2)

  return A2 > 0.5
```

#### 4.5.2 Evaluation.

The ```evaluate``` function computes the accuracy of the model, the output will be the percent accuracy that the model performance compare to the ground truth $(Y)$.


```python
%%add_to OneHiddenLayerNN
def evaluate(self, X, Y):
  """
  Evaluate model accuracy

  Arguments:
  X -- input data of size (self.n_input_size, m)
  Y -- true label vector with dim (1, number of example)
  """

  Z1 = np.dot(self.W1, X) + self.b1
  A1 = np.tanh(Z1)
  Z2 = np.dot(self.W2, A1) + self.b2
  A2 = self.sigmoid(Z2)

  result = {"accuracy": (100 - np.mean(np.abs(A2 - Y))*100)}
  return result
```

## 5. Test the Model

Let's train the model and see how it compares to the ealier Logistic Regression model on our dataset.


```python
# Define layer units
n_input_layer  = X.shape[0]
n_hidden_layer = 5
n_output_layer = Y.shape[0]

nn_model = OneHiddenLayerNN(n_input_layer, n_hidden_layer, n_output_layer)
nn_model.train(X, Y, n_iterations=5000, learning_rate=1.2, print_log=True)
nn_model.evaluate(X,Y)
```

    Iteration: 0 - Cost: 0.693114
    Iteration: 1000 - Cost: 0.285105
    Iteration: 2000 - Cost: 0.070895
    Iteration: 3000 - Cost: 0.053353
    Iteration: 4000 - Cost: 0.050997
    
    {'accuracy': 97.24793573791678}



* The model's accuracy is now 97%, compare to 87% from Logistic Regression model. 

* Let's plot the decision boundary to see how our neural network model capture the moon shape of the dataset.


```python
plot_decision_boundary(lambda x: nn_model.predict(x.T), X, Y)
```


    
![png](\assets\images\blogs\2022-11-07-shallow-neural-network-with-one-hidden-layer-6.png)
    


As can be seen, the neural network model learned the patterns of the moon shape while the Logistic Regression model did not. This means that neural networks can learn non-linear decision boundaries.

## 6. Test Model with different layer size


```python
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 30, 70]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    nn_model = OneHiddenLayerNN(X.shape[0],n_h,1)
    nn_model.train(X, Y, n_iterations=5000, learning_rate=1.2)
    plot_decision_boundary(lambda x: nn_model.predict(x.T), X, Y)
    predictions = nn_model.predict(X)
    evaluations = nn_model.evaluate(X, Y)
    print ("Accuracy for {} hidden units: {} %".format(n_h, float(evaluations['accuracy'])))
```

    Accuracy for 1 hidden units: 67.5415405706139 %
    Accuracy for 2 hidden units: 81.26683852538964 %
    Accuracy for 3 hidden units: 89.09724444638869 %
    Accuracy for 4 hidden units: 94.12905351582852 %
    Accuracy for 5 hidden units: 95.13494340238464 %
    Accuracy for 20 hidden units: 96.5724858704476 %
    Accuracy for 30 hidden units: 94.20160820644769 %
    Accuracy for 70 hidden units: 94.56447128630427 %



    
![png](\assets\images\blogs\2022-11-07-shallow-neural-network-with-one-hidden-layer-7.png)
    


* The more hidden layer units, the better it fits the dataset.

* When the hidden layer units is too large, the model tends to overfit the patterns of the dataset.

