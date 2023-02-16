---
layout: post
comments: false
title: Errors in Hypothesis Testing
categories: [Fundamental Concepts of Hypothesis Testing]
---

Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and variance $\sigma^2=2$.

$$
  H_0:\mu\le3\qquad H_1:\mu\gt3
$$

Idea: Look at $\overline{X}$ and reject $H_0$ in favor of $H_1$ if $\overline{X}$ is "large".

i.e. Look at $\overline{X}$ and reject $H_0$ in favor of $H_1$ if $\overline{X}\gt c$ for some value $c$.

**Errors in Hypothesis Testing**

![png](\assets\images\notes\errors-in-hypothesis-testing.png)

* Type I or Type II error which is worse?
* This totally depends on how you set up hypotheses and what is at stake.
* The null hypothesis is assumed to be true and the alternate hypothesis is what you are out to show.

<font color='green'>Example</font>

1. You are a potato chip manufacturer and you want to ensure that the mean amount in 15 ounce bags is at least 15 ounces.

  $$
    H_0:\mu\le15\\
    H_1:\mu\gt15
  $$

2. You are an angry consumer group and you want to show that the chip company is cheating its customers.

  $$
    H_0:\mu\ge15\\
    H_1:\mu\lt15
  $$

These examples we out to show the alternate hypothesis. Back to example 1, we have:

<font color='red'>Type I error:</font>

The true mean is $\le15$ but you concluded it was $\gt15$. You are going to save some money because you won't be adding chips but you are risking a lawsuit!

<font color='red'>Type II error:</font>

The true mean is $\gt15$ but you concluded it was $\le15$. You are going to be spending money increasing the amount of chips when you didn't have to.



