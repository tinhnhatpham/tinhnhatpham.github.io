---
layout: post
comments: false
title: Jointly Distributed Random Variables
categories: [Joint Distributions and Covariance]
---

### **Discrete Random Variables**

**Example:** An insurance agency services customers who have
both a homeowner's policy and an automobile policy. For each
type of policy, a deductible amount must be specified. For an
automobile policy, the choices are \\\$100 or \\\$250 and for the
homeowner's policy, the choices are \\\$0, \\\$100, or \\\$200.

Suppose an individual, let’s say Bob, is selected at random

from the agency’s files. Let $X$ be the deductible amount on
the auto policy and let Y be the deductible amount on the
homeowner's policy.

We want to understand the relationship between $X$ and $Y$.

Suppose the **joint probability table** is given by the insurance company as follows:

![png](\assets\images\notes\2022-07-15-jointly-distributed-random-variables-1.png)

$$
  P(X=100, Y=0)=0.2\\
  P(X=250,Y=0)=0.05\\
  \implies P(Y=0)=0.25
$$

**Definition:** Given two discrete random variables, $X$ and $Y$, $p(x,y)=P(X=x,Y=y)$ is the **joint probability mass function** for $X$ and $Y$.

![png](\assets\images\notes\2022-07-15-jointly-distributed-random-variables-2.png)

Recall: Two events, $A$ and $B$, are independent if $P(A\cap B)=P(A)P(B)$.

In insurance example: are $X$ and $Y$ independent?

$$
  \begin{align}
    &P(X=100,Y=100)=0.1\\
    &P(X=100)P(Y=100)=(0.5)(0.25)=1.25\\
    &\implies X\text{ and }Y\text{ are not indept.}
  \end{align}
$$

<u>Important property:</u> $X$ and $Y$ are **independent random variables** if $P(X=x,Y=y)=P(X=x)P(Y=y)$ for all possible values of $x$ and $y$.

---

### **Continuous Random Variables**

**Definition:** if $X$ and $Y$ are continuous random variables, then $f(x,y)$ is the **joint probability density function** for $X$ and $Y$ if

$$
  P(a\le X\le b,c\le Y\le d)=\int_{a}^{b}\int_{c}^{d}f(x,y)dxdy
$$

for all possible $a$, $b$, $c$, and $d$.

<u>Important property:</u> $X$ and $Y$ are **independent random variables** if $f(x,y)=f(x)f(y)$ for all possible values of $x$ and $y$.

**Example:** Suppose a room is lit with two light bulbs. Let $X_1$ be the lifetime of the first bulb and $X_2$ be the lifetime of the second bulb. Suppose $X_1\sim\text{Exp}(\lambda_1=1/2000)$ and, $X_2\sim\text{Exp}(\lambda_2=1/3000)$. If we assume the lifetimes of the light bulbs are independent of each other, find the probability that the room is dark after 4000 hours.

$$
  E(X_1)=\frac{1}{\lambda_1}=2000\text{ hours}\\
  E(X_2)=\frac{1}{\lambda_2}=3000\text{ hours}\\
$$

Light bulbs function independently, so

$$
  \begin{align}
    P(X_1\le4000,X_2\le4000)&=P(X_1\le4000)P(X_2\le4000)\\
    &=\bigg(\int_0^{4000}\lambda_1e^{-\lambda_1x_1}dx_1\bigg)\bigg(\int_0^{4000}\lambda_2e^{-\lambda_2x_2}dx_2\bigg)\\
    &=\bigg(-e^{-\lambda_1x_1}\bigg)\Bigg\rvert_{0}^{4000}\bigg(-e^{-\lambda_2x_2}\bigg)\Bigg\rvert_{0}^{4000}\\
    &=\bigg(1-e^{-4000/2000}\bigg)\bigg(1-e^{-4000/3000}\bigg)\\
    &=(1-e^{-2})(1-e^{-4/3})\\
    &\approx0.6368
  \end{align}
$$