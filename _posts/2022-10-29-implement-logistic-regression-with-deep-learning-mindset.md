---
layout: post # You can ommit this if you've set it as a default
title: Implement Logistic Regression with Deep Learning Mindset
categories: [Deep Learning]
comments: true
---

In this exercise, we will implement a simple Logistic Regression model using neural network mindset (forward and backward propagation).

Logistic Regression is a supervised learning classification algorithm used to predict the probability of a target variable. In this classification, there would be only two possible classes.

For this implementation, we use Logistic Regression to classify cat and dog pictures (1 - cat vs. 0 - dog).

## Data Processing

We use the subset of ["Dogs vs. Cats" dataset](https://www.kaggle.com/c/dogs-vs-cats/data) available on Kaggle, which contains 25,000 images.

* For the purpose of this exercise, we only use 500 images of dogs and cats for our traing data to reduce the training time.

* The validation data will have 200 images of dogs and cats.


```python
# Download the dataset

%%capture
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip
```


```python
# Import all the packages

import os
import zipfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Extract the dataset
local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
```

Get all the do and cats directories.


```python
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
```

The images will be converted to numpy array:

* The numpy array with dimension (number of images, number width pixels, number of height pixels).

* All the images will be resize to 128x128 pixels. This is important, we need to keep the dimensions consistently to eliminate bugs when feeding them into the model.


```python
def img_to_array(folder_path, resize_shape = None, n_images = 1000):
  """
  Covert image to numpy array with shape (num_images, num_px_width * num_px_height *num_channel)

  Arguments:
  folder_path -- path to folder that contains images
  resize_shape -- resize image to the desired shape
  n_images -- number of images to be converted

  Returns:
  arr - numpy array with shape (num_images, num_px_width * num_px_height *num_channel)
  """

  file_names = os.listdir(folder_path)
  img_array = []
  
  for i in range(len(file_names)):
    file_name = file_names[i]
    
    # Take only  n_images for each type
    if i >= n_images:
      break

    img = Image.open(os.path.join(folder_path, file_name))
    if resize_shape != None:
      img = img.resize(resize_shape)

    img_array.append(np.asarray(img))
  return np.asarray(img_array, dtype=np.float128)

# Resize images to the same size (128x128)
resize = (128,128)

# Convert images to numpy array
train_cat_images = img_to_array(train_cats_dir, resize, 250) 
train_dog_images = img_to_array(train_dogs_dir, resize, 250) 
val_cat_images = img_to_array(validation_dogs_dir, resize, 100) 
val_dog_images = img_to_array(validation_cats_dir, resize, 100) 

# Create labels for dataset, 1 for cat and 0 for dog
train_cat_labels = np.ones((train_cat_images.shape[0],1))
train_dog_labels = np.zeros((train_dog_images.shape[0],1))
val_cat_labels = np.ones((val_cat_images.shape[0],1))
val_dog_labels = np.zeros((val_dog_images.shape[0],1))
```

Let's take a look at our training set to get a better sense of what dogs and cats datasets look like. We will display 8 images for each type.


```python
# Create figure
fig = plt.figure(figsize=(16,16))

# Configure display parameters
rows = 4
cols = 4
total = rows*cols

for i in range(int(total/2)):
  fig.add_subplot(rows, cols, i + 1)
  plt.imshow(train_cat_images[i].astype(np.uint8))
  plt.title("Cat")
  plt.axis('off')

  fig.add_subplot(rows, cols, int(total/2) + i + 1)
  plt.imshow(train_dog_images[int(total/2) + i].astype(np.uint8))
  plt.title("Dog")
  plt.axis('off')

```


    
![png](\assets\images\blogs\2022-10-29-logistic-regression-with-deep-learning-mindset-1.png)
    


Check again the shape of datasets.


```python
print("train_cats_images shape", train_cat_images.shape)
print("train_cat_labels shape", train_cat_labels.shape)
print()
print("train_dogs_images shape", train_dog_images.shape)
print("train_dog_labels shape", train_dog_labels.shape)
print()
print("val_cats_images shape", val_cat_images.shape)
print("val_cat_labels shape", val_cat_labels.shape)
print()
print("val_dogs_images shape", val_dog_images.shape)
print("train_dog_labels shape", train_dog_labels.shape)
```

    train_cats_images shape (250, 128, 128, 3)
    train_cat_labels shape (250, 1)
    
    train_dogs_images shape (250, 128, 128, 3)
    train_dog_labels shape (250, 1)
    
    val_cats_images shape (100, 128, 128, 3)
    val_cat_labels shape (100, 1)
    
    val_dogs_images shape (100, 128, 128, 3)
    train_dog_labels shape (250, 1)
    

Next step, we put cats and dogs datasets together and shuffle them.


```python
train_set_x = np.concatenate((train_cat_images, train_dog_images))
train_set_y = np.concatenate((train_cat_labels, train_dog_labels))
val_set_x = np.concatenate((val_cat_images, train_dog_images))
val_set_y = np.concatenate((val_cat_labels, train_dog_labels))

from sklearn.utils import shuffle

# Shuffle the joint dataset
train_set_x, train_set_y = shuffle(train_set_x, train_set_y, random_state=0)
val_set_x, val_set_y = shuffle(val_set_x, val_set_y, random_state=0)
```

![png](\assets\images\blogs\2022-10-29-logistic-regression-with-deep-learning-mindset-2.png)

The training and validation datasets has three channels (red, green, and blue) for each image. We make it flat so that each pixel is a feature. Each flattened image was also stacked vertically, as shown in the image above.


```python
# Flatten images and set the shape to (n_features, m_set)
X_train = train_set_x.reshape(train_set_x.shape[0], -1).T
Y_train = train_set_y.T

X_val = val_set_x.reshape(val_set_x.shape[0], -1).T
Y_val = val_set_y.T

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_val shape: ", X_val.shape)
print("Y_val shape: ", Y_val.shape)
```

    X_train shape:  (49152, 500)
    Y_train shape:  (1, 500)
    X_val shape:  (49152, 350)
    Y_val shape:  (1, 350)
    

We standardize the datasets before feeding to the neural network (model). For image datasets, all the images are in range [0,255], so we only need to divide the images to 255 (the maximum value of a pixel).


```python
# Standardize by divide the dataset by 255
X_train = X_train / 255.
X_val = X_val / 255.
```

## Logistic Regression Model Implemetation

### Model Architecture 

We will build a Logistic Regression model using a Neural Network mindset. This is a simple neural network shown as below.

![png](\assets\images\blogs\2022-10-29-logistic-regression-with-deep-learning-mindset-3.png)

The figure above is a mathematical expression of the algorithm with one example $x^{(i)}$. We have 

$$
    \begin{align}
        z^{(i)}&=w^Tx^{(i)}+b\\
        \widehat{y}^{(i)}&=a^{(i)}=\sigma\left(z^{(i)}\right)\qquad(\sigma=\text{sigmoid})\\
        \mathcal{L}\left(a^{(i)},y^{(i)}\right)&=-y^{(i)}\log\left(a^{(i)}\right)-\left(1-y^{(i)}\right)\log\left(1-a^{(i)}\right)\\
    \end{align}
$$

with

$$
    \begin{align}
        &z^{(i)}&&\text{a linear equation with weight $w$ and bias $b$}\\
        &\widehat{y}^{(i)},a^{(i)}&&\text{result of sigmoid function apply to $z^{(i)}$, (estimated value)}\\
        &\mathcal{L}\left(a^{(i)},y^{(i)}\right)&&\text{cost function between estimated value and ground truth}\\
    \end{align}
$$

The cost is the computed by averaging all costs of entire set:

$$
    J=\frac{1}{m}\sum_{i=1}^m\mathcal{L}\left(a^{(i)},y^{(i)}\right)
$$

We will talk more about cost function in forward propagation later.

### Building Parts of The Algorithm

#### 1. Sigmoid Function

In machine learning, the term sigmoid function is normally used to refer specifically to the logistic function, also called the logistic sigmoid function. The use of a sigmoid function is to convert a real value into one that can be interpreted as a probability

All sigmoid functions have the property that they map the entire number line into a small range such as between 0 and 1, or -1 and 1, so one use of a sigmoid function is to convert a real value into one that can be interpreted as a probability. ([Click for more](https://deepai.org/machine-learning-glossary-and-terms/sigmoid-function))

$$
    \sigma(z)=\frac{1}{1 + e^{-z}}
$$


```python
# sigmoid

def sigmoid(z):
  """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar of numpy array 

    Return:
    s -- sigmoid of z
  """

  return 1/(1 + np.exp(-z))

```

#### 2. Parameters Initialization 

We will initialize weight $w$ as a vector of zeros with shape (num_of_feature, 1) and bias $b$ with zero.


```python
# Initialize paramenters
def parameters_initialize(dim):
  """
  Create a vector of zeroes with the shape (dim, 1) for parameter w and initializes b to 0

  Argument:
  dim -- size of the w vector

  Returns:
  w -- initialized vector of zeroes with the shape (dim, 1)
  b -- initialized scalar (bias) with float type
  """

  w = np.zeros((dim, 1))
  b = 0.0
  return w, b
```

#### 3. Forward and Backward Propagation

**3.1 Forward Propagation**

In forward propagation, we will implement these mathematical expressions:

$$
    \begin{align}
        z&=w^TX+b\\
        \widehat{y}&=a=\sigma\left(z\right)\\
    \end{align}
$$

In logistic regression, we don't use MSE L2 (Mean Square Error) as in linear regression. Because the model predictions is non-linear (sigmoid transformation). If we use MSE $\mathcal{L}\left(a^{(i)},y^{(i)}\right)=1/2(\widehat{y}-y)^2$, the result is a non-convex funtion with many local minimums. Our gradient descent might not find the optimal global minimum.

Instead, we use a cost function called **Cross-entropy** as shown below:

$$
    \mathcal{L}\left(\widehat{y},y\right)=-\left(y\log\widehat{y}+(1-y)\log(1-\widehat{y})\right)
$$

We want the cost to be optimized as small as possible so that $y$ and $\widehat{y}$ close together.

$$
    \begin{cases}
        \begin{align}
            &y=1&&-\mathcal{L}\left(\widehat{y},y\right)=-\log\widehat{y}&&&\leftarrow\widehat{y}\text{ need to be large}\\
            &y=0&&-\mathcal{L}\left(\widehat{y},y\right)=-\log(1-\widehat{y})&&&\leftarrow\widehat{y}\text{ need to be small}\\
        \end{align}
    \end{cases}
$$

The cost is the computed by averaging all costs of entire set:

$$
    \begin{align}
        J(w,b)&=\frac{1}{m}\sum_{i=1}^m\mathcal{L}\left(\widehat{y}^{(i)},y^{(i)}\right)\\
        &=-\frac{1}{m}\sum_{i=1}^my^{(i)}\log\widehat{y}^{(i)}+(1-y^{(i)})\log(1-\widehat{y}^{(i)})
    \end{align}
$$


**3.2 Backward Propagation**

We have cost function from the forward propagation. To minimize it, we use Gradient Descent to update $w$ and $b$ until we find the global minimum (minimum cost).

![png](\assets\images\blogs\2022-10-29-logistic-regression-with-deep-learning-mindset-4.png)

To do backward propagation, we're going to find the partial derivative of $\mathcal{L}$ w.r.t $w$ and $b$. It means we need to find:

$$
    \frac{\partial{\mathcal{L}}}{\partial{w}}\quad\text{and}\quad\frac{\partial{\mathcal{L}}}{\partial{b}}
$$

By the Chain Rule in Calculus

$$
    \begin{align}
        \frac{\partial{\mathcal{L}}}{\partial{w}}&=\frac{\partial{\mathcal{L}}}{\partial{a}}\times\frac{\partial{a}}{\partial{z}}\times\frac{\partial{z}}{\partial{w}}\\
        \\
        \frac{\partial{\mathcal{L}}}{\partial{b}}&=\frac{\partial{\mathcal{L}}}{\partial{a}}\times\frac{\partial{a}}{\partial{z}}\times\frac{\partial{z}}{\partial{b}}\\
    \end{align}
$$

The cost function is

$$
    \mathcal{L}=-y\log(a)+(1-y)\log(1-a)
$$

So

$$
    \begin{align}
        \frac{\partial{\mathcal{L}}}{\partial{a}}&=-y\frac{1}{a}-(1-y)\frac{1}{1-a}-1\\
        &=\frac{-y}{a}+\frac{1-y}{1-a}\\
        &=\frac{-y(1-a)+a(1-y)}{a(1-a)}\\
        &=\frac{a-y}{a(1-a)}\qquad(1)
    \end{align}
$$

We know that derivative of a sigmoid is:

$$
    \frac{\partial{\sigma(z)}}{\partial{z}}=\sigma(z)(1-\sigma(z))\\
    \implies \frac{\partial{a}}{\partial{z}}=a(1-a)\qquad(2)\\
$$

$$
    \begin{align}
        \frac{\partial{z}}{\partial{w}}&=x\qquad(3)\\
        \frac{\partial{z}}{\partial{b}}&=1\qquad(4)\\
    \end{align}
$$

Now we can calculate the partial derivative of $\mathcal{L}$ w.r.t $w$ and $b$

$$
    \begin{align}
        \color{red}{\frac{\partial{\mathcal{L}}}{\partial{w}}}&=(1)\times(2)\times(3)\\
        &=\frac{a-y}{a(1-a)}\times a(1-a)\times x\\
        &=\color{red}{(a-y)x}\\
        \\
        \color{red}{\frac{\partial{\mathcal{L}}}{\partial{b}}}&=(1)\times(2)\times(4)\\
        &=\frac{a-y}{a(1-a)}\times a(1-a)\times 1\\
        &=\color{red}{(a-y)}
    \end{align}
$$

To perform an update for $w$ and $b$ we do the following steps

$$
    w:=w-\alpha\frac{\partial{\mathcal{L}}}{\partial{w}}\\
    b:=b-\alpha\frac{\partial{\mathcal{L}}}{\partial{b}}\\
    (\alpha\text{ is learning rate})
$$

We have all we need to implement forward and backward propagation.


```python
# Forward and backward propagation

def propagate(X, Y, w, b):
  """
  Implement cost function and its gradient for forward and backward propagation

  Arguments:
  X -- training data with dim (num_px_width * num_px_height *num_channel, number of examples)
  Y -- true label vector (1 if cat, 0 if dog) with dim (1, number of example)
  w -- weights, a numpy array with size (num_px_width * num_px_height *num_channel, 1)
  b -- bias, a scalar

  Return:
  cost -- log-likelihood cost 
  dw -- gradient of the loss w.r.t w (same shape as w)
  db -- gradient of the loss w.r.t b (same shape as b)
  """
  m = X.shape[1]

  # FORWARD PROPAGATION
  A = sigmoid(np.dot(w.T, X) + b)

  # Add a very small epsilon to avoid log(0)
  epsilon = 1e-5    
  cost = -1/m * np.sum((Y*np.log(A + epsilon) + (1 - Y)*np.log(1 - A + epsilon)))

  # BACWARD PROPAGATION
  # divide by m because the cost J was computed by averaging the entire set.
  dw = 1/m * np.dot(X, (A - Y).T)
  db = 1/m * np.sum(A - Y)

  cost = np.squeeze(np.array(cost))

  grads = {"dw":dw, "db": db}
  return cost, grads

```

The ```propagate``` function computes the cost and gradient $\partial{w}$ and $\partial{b}$ for 1 iteration through entire training examples.

To find the global minimum of the cost, we will loop through $n$ interations and update $w$ and $b$ by $\partial{w}$ and $\partial{b}$, the ```optimize``` function below will propagate through training examples $n$ times.


```python
def optimize(X, Y, w, b, n_iterations = 100, learning_rate = 0.001, print_log=False):
  """
  Update w and b using gradient descent

  Arguments:
  X -- training data with dim (num_px_width * num_px_height *num_channel, number of examples)
  Y -- true label vector (1 if cat, 0 if dog) with dim (1, number of example)
  w -- weights, a numpy array with size (num_px_width * num_px_height *num_channel, 1)
  b -- bias, a scalar
  n_iterations -- number of iterations through the data examples
  learning_rate -- step size for each iteration
  print_log -- print the loss every 100 steps

  Returns:
  params -- dictionary contains w and b
  grads -- dictionary contains dw and db w.r.t to the cost function
  costs -- list all costs during training
  """

  # Make a deep copy before update w and b
  w = copy.deepcopy(w)
  b = copy.deepcopy(b)

  costs = []

  for i in range(n_iterations):
    cost, grads = propagate(X, Y, w, b)
    # Retrieve dw and db
    dw = grads['dw']
    db = grads['db']

    # Update parameters w and b
    w = w - learning_rate*dw
    b = b - learning_rate*db

    # Record cost and print cost every 100 interations
    if i%500 == 0:
      costs.append(cost)

      if print_log:
          print ("Iteration: %i - Cost: %f" %(i, cost))
    
    
  params = {
      'w': w,
      'b': b
  }

  grads = {
      'dw': dw,
      'db': db,
  }

  return params, grads, costs
```

So now we use the optimized $w$ and $b$ to get predictions from the model.


```python
def predict(X, w, b):
  """
  Predict data X with label 1 (cat) or 0 (dog) using learn parameters w and b

  Arguments:
  X -- data with dim (num_px_width * num_px_height *num_channel, number of examples)
  w -- weights, a numpy array with size (num_px_width * num_px_height *num_channel, 1)
  b -- bias, a scalar

  Return:
  Y_predictions -- a numpy array contains predictions for the data X
  """
  
  A = sigmoid(np.dot(w.T, X) + b)

  Y_predictions = A > 0.5

  return Y_predictions

```

### Build a Logistic Regression Model by merging all functions

#### 1. Merging all functions

Now we have the overall structure of the model, it's time to put them all together (all the functions implemented in the building parts).


```python
class LogisticRegressionModel():
  """
  Regression Logistic Model using forward and backward propagation.
  """

  def __init__(self):
    # w: weights, a numpy array with size (num_px_width * num_px_height *num_channel, 1)
    self.w = None
    # b: bias, a scalar
    self.b = None

    self.costs = []
    self.learning_rate = 0.001
    self.n_iterations = 1000
    self.print_log = False

  def parameters_initialize(self, dim):
    """
    Create a vector of zeroes with the shape (dim, 1) for parameter w and initializes b to 0

    Argument:
    dim -- size of the w vector

    Returns:
    w -- initialized vector of zeroes with the shape (dim, 1)
    b -- initialized scalar (bias) with float type
    """

    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

  def propagate(self, X, Y):
    """
    Implement cost function and its gradient for forward and backward propagation

    Arguments:
    X -- training data with dim (num_px_width * num_px_height *num_channel, number of examples)
    Y -- true label vector (1 if cat, 0 if dog) with dim (1, number of example) 

    Returns:
    cost -- log-likelihood cost 
    dw -- gradient of the loss w.r.t w (same shape as w)
    db -- gradient of the loss w.r.t b (same shape as b)
    """
    m = X.shape[1]

    # Forward propagation
    A = sigmoid(np.dot(self.w.T, X) + self.b)

    # Add a very small epsilon to avoid log(0)
    epsilon = 1e-5    
    cost = -1/m * np.sum((Y*np.log(A + epsilon) + (1 - Y)*np.log(1 - A + epsilon)))

    # Backward propagation
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw":dw, "db": db}
    return cost, grads

  def optimize(self, X, Y):
    """
    Update w and b using gradient descent

    Arguments:
    X -- training data with dim (num_px_width * num_px_height *num_channel, number of examples)
    Y -- true label vector (1 if cat, 0 if dog) with dim (1, number of example)
    """

    for i in range(self.n_iterations):
      cost, grads = self.propagate(X, Y)
      # Retrieve dw and db
      dw = grads['dw']
      db = grads['db']

      # Update parameters w and b
      self.w = self.w - self.learning_rate*dw
      self.b = self.b - self.learning_rate*db

      # Record cost and print cost every 100 interations
      if i%500 == 0:
        self.costs.append(cost)

        if self.print_log:
            print ("Iteration: %i - Cost: %f" %(i, cost))
  
  def predict(self, X):
    """
    Predict data X with label 1 (cat) or 0 (dog) using learn parameters w and b

    Arguments:
    X -- data with dim (num_px_width * num_px_height *num_channel, number of examples)
    
    Return:
    Y_predictions -- a numpy array contains predictions for the data X
    """
    
    A = sigmoid(np.dot(self.w.T, X) + self.b)

    Y_predictions = A > 0.5

    return Y_predictions
  
  def train(self, X_train, Y_train, n_iterations = 2000, learning_rate = 0.001, print_log=False):
    """
    Build regression logistic model

    Arguments:
    X_train -- training data with dim (num_px_width * num_px_height *num_channel, number of examples)
    Y_train -- true training label vector (1 if cat, 0 if dog) with dim (1, number of example)
    n_iterations -- number of iterations through the data examples
    learning_rate -- step size for each iteration
    print_log -- print the loss every n steps
    """
    
    # Initialize parameter w and b
    self.w, self.b = self.parameters_initialize(X_train.shape[0])

    self.learning_rate = learning_rate
    self.n_iterations = n_iterations
    self.print_log = print_log

    # Run forward and backward propagation to optimze w and b
    self.optimize(X_train, Y_train)

  def evaluate(self, X, Y):
    preds = self.predict(X)
    result = {"accuracy": (100 - np.mean(np.abs(preds - Y))*100)}
    return result

```

#### 2. Train and Evaluate Model

Let's train our processed dataset in previous section.


```python
model = LogisticRegressionModel()
model.train(X_train, Y_train, n_iterations= 3000, learning_rate=0.001, print_log=True)
```

    Iteration: 0 - Cost: 0.693127
    Iteration: 500 - Cost: 0.623720
    Iteration: 1000 - Cost: 0.418507
    Iteration: 1500 - Cost: 0.346329
    Iteration: 2000 - Cost: 0.306877
    Iteration: 2500 - Cost: 0.275673
    

Evaluate both training set and validation set.


```python
print(model.evaluate(X_val, Y_val))
print(model.evaluate(X_train, Y_train))
```

    {'accuracy': 81.42857142857143}
    {'accuracy': 97.0}
    

Training accuracy is 97% and validation accuracy is 81%.

* The model is just a simple neural network, but it has high enough capacity to fit the training data.

* The model's accuracy for training data is close to 100%. That means our model is overfitting. When evaluate with validation data, the accuracy dropped to 81%.

Let plot the learning curve with the cost function and the gradients.


```python
plt.plot(model.costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 500 hundreds)')
plt.title("Learning rate =" + str(model.learning_rate))
plt.show()
```


    
![png](\assets\images\blogs\2022-10-29-logistic-regression-with-deep-learning-mindset-5.png)
    


We can see that the cost is declining, and we could train the model more accurately by running more iterations. This occurs as a result of the model's attempt to match the parameters to this specific training set of data. But when evaluating with unforeseen data, accuracy will drop. Overfitting is what we refer to as in this situation.
