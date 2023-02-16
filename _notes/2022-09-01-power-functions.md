---
layout: post
comments: false
title: Power Functions
categories: [Composite Test - Power Functions - and P-Values]
---

Let $X_1,X_2,...,X_n$ be a random sample from any distribution with unknown parameter $\theta$ which takes values in a parameter space $\Theta$.

We ultimately want to test

$$
  H_0:\theta\in\Theta_0\\
  H_1:\theta\in\Theta \backslash\Theta_0
$$

where $\Theta_0$ is some subset of $\Theta$.

**The Power Function**

$$
  \begin{align}
    \gamma(\theta)&=P(\text{Reject $H_0$ when the parameter is $\theta$})\\
    &=P(\text{Reject $H_0;\theta$})
  \end{align}
$$

$\theta$ is an argument that can be anywhere in the parameter space $\Theta$.

* It could be a $\theta$ from $H_0$.
* It could be a $\theta$ form $H_1$.

<u>Note that</u>

For $\alpha$:

$$
  \begin{align}
    \alpha&=\text{max}P(\text{Reject $H_0$ when true})\\
    &=\underset{\theta\in\Theta_0}{\text{max}}P(\text{Reject $H_0;\theta$})\\
    &=\underset{\theta\in\Theta_0}{\text{max}}\gamma(\theta)\\
    &\big(\text{Other notation is $\underset{\theta\in H_0}{\text{max}}$}\big)
  \end{align}
$$

For $\beta$:

$$
  \begin{align}
    \beta&=\text{max}P(\text{Fail to reject $H_0$ when false})\\
    &=\underset{\theta\in\Theta\backslash\Theta_0}{\text{max}}P(\text{Fail to Reject $H_0;\theta$})\\
    &=\underset{\theta\in\Theta\backslash\Theta_0}{\text{max}}\bigg[1 - P(\text{Reject $H_0;\theta$})\bigg]\\
    &=\underset{\theta\in\Theta\backslash\Theta_0}{\text{max}}\bigg[1-\gamma(\theta)\bigg]\\
    &\big(\text{Other notation is $\underset{\theta\in H_1}{\text{max}}$}\big)
  \end{align}
$$

Why do we use power function?

* They are great for comparing two hypothesis tests.