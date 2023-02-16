---
layout: post
comments: false
title: Discrete Random Variable
categories: [Discrete Random Variable]
---

### **Random Variable**

**Definition:** A random variable (rv) is a function that maps events (from the sample space $S$) to the real numbers.

$$
  X:S\rightarrow \mathbb{R}
$$

Random variables can be **discrete** or **continuous**, or sometimes a mixture of the two.

A rv is **discrete** if its set of possible values is discrete.

Example: Flip a fair coin 100 times. Let $Y$ be the number of heads in the 100 flips. $Y$ can take values $0,1,2,\ldots,100$.

A rv is **continuous** if its set of possible values is an entire interval of numbers.

Example: Let $T$ bet the time between two customers entering a store. Let $U$ be a random number from the interval $[-5,5]$.

<u>Convention:</u> We usually denote radom variables by a capital letter near the end of the alphabet (e.g. $X$, $Y$) and specific instances of the random variable by a lower case letter.

$$
  \underbrace{X}_{r.v.}= \underbrace{x}_{\text{realization of r.v.}}
$$

**Big Picture** In statistics, we will model populations using random variables, and features/parameters (e.g. mean, variance) of these random variables will tell us about the population we are studying.

### **Probability Mass Function (pmf)**

Experiment: Roll a six-sided dice twice.

$S=\{(i,j)\mid i,j\in\{1,2,3,4,5,6\}\}$. Let $X$ be the sum of the two rolls. $X$ can take values $2,3,\ldots,12$.

$$
  \begin{align}
    P(X=2)&=P(\{11\})=\frac{1}{36}=P(\{66\})=P(X=12)\\
    P(X=3)&=P(\{12,21\})=\frac{2}{36}=P(\{56,65\})=P(X=11)\\
    P(X=4)&=\frac{3}{36}=P(X=10)\\
    P(X=5)&=\frac{4}{36}=P(X=9)\\
    P(X=6)&=\frac{5}{36}=P(X=8)\\
    P(X=7)&=P(\{16,25,34,43,52,61\})=\frac{6}{36}\\
  \end{align}
$$

$$
  \sum_{k=2}^{12}P(X=k)=1
$$

**Definition:** A probability mass function of a discrete rv, $X$, is given by

$$
  \color{red}{(x)=P(X=x)=P(\text{all x}\in S\mid X(s)=x)}
$$

What do we expect?

1. $0\le P(X=x)\le 1\leftarrow$ Axiom 1
2. $\sum_{x}P(X=x)=P(S)=1\leftarrow$ Axiom 2
3. $P(X=a)\cup P(X=b)=P(X=a)+P(X=b)\leftarrow$ Axiom 3

**Example**

A lab has 6 computers. Let $X$ denote the number of these computers that are in use during the lunch hour. Suppose the pmf of $X$ is given by:

| x            | 0    | 1    | 2    | 3    | 4    | 5    | 6   |
|--------------|------|------|------|------|------|------|-----|
| P(X=x)= p(x) | 0.05 | 0.10 | 0.15 | 0.25 | 0.20 | 0.15 | 0.1 |

Probability that at most 2 computers are in use:

$$
  \begin{align}
    P(X\le2)&=P(X=0)+P(X=1)+P(X=2)\\
    &=0.05+0.10+0.15=0.3
  \end{align}
$$

Probability that at least haft of the computers are in use

$$
  \begin{align}
    P(X\ge3)&=P(X=3)+P(X=4)+P(X=5)+P(X=6)\\
    &=0.7\\
    P(X\ge3)&=1-P(X\le2)
  \end{align}
$$

Probability that there are 3 or 4 computers free:

$$
  P(X=3\cup X=2)=P(X=3)+P(X=2)=0.4
$$

### **Cumulative distribution function (cdf)**

The **cumulative distribution function (cdf)** is given by

$$
  \color{red}{F(y)=P(X\le y)=\sum_{x\le y}P(X=x)}
$$

| x            | 0    | 1    | 2    | 3    | 4    | 5    | 6   |
|--------------|------|------|------|------|------|------|-----|
| P(X=x)= p(x) | 0.05 | 0.10 | 0.15 | 0.25 | 0.20 | 0.15 | 0.1 |

For discrete r.v., cdf is a step function.

$$
  F(y)=\begin{cases}
    \begin{align}
      &0&&\text{if }y\lt0\\
      &0.05&&\text{if }0\le y\lt1\\
      &0.15&&\text{if }1\le y\lt2\\
      &0.30&&\text{if }2\le y\lt3\\
      &0.55&&\text{if }3\le y\lt4\\
      &0.75&&\text{if }4\le y\lt5\\
      &0.9&&\text{if }5\le y\lt6\\
      &1&&\text{if }6\le y\\
    \end{align}
  \end{cases}
$$