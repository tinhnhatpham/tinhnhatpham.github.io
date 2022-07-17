---
layout: post
title: The Poisson and Exponential Random Variables
categories: [Probability]
---
**The Poisson random variable**
A Poisson rv is a discrete rv that describes the total number of events that happen in a certain time period.
* **Definition:** A discrete random variable X has **Poisson distribution** with parameter $\lambda (\lambda \gt 0)$ if the probability mass function of $X$ is:

* $\text{PMF:} \quad P(X = k) = \frac{\lambda^k}{k!}e^{-\lambda} \quad \text{for k = 0, 1, 2...}$
* $E(X) = \lambda$
* $E(X^2) = \lambda(\lambda + 1)$
* $V(X) = E(X^2) - (E(X))^2 = \lambda$
* **Notation:** $X \sim Poisson(\lambda)$

**The exponential random variable**
The family of exponential distributions provides probability models that are widely used in engineering and science disciplines to describe **time-to-event data**. An exponential rv is continous. Example: Time until birth, time until light bulb fails, waiting time in a queue, length of service time, time between customer arrivals.
* **Definition:** A continous random variable $X$ has the exponential distribution with rate parameter $\lambda (\lambda \gt 0)$ if the pdf of $X$ is:

$$
  f(x)=\begin{cases}
    \lambda e^{-\lambda x} & \text{if $x \ge 0$}.\\
    0, & \text{else}.
  \end{cases}
$$

* $E(X) = \frac{1}{\lambda}$
* $E(X^2) = \frac{2}{\lambda^2}$
* $V(X) = E(X^2) - (E(X))^2 = \frac{1}{\lambda^2}$
* **Notation:** $X \sim Exp(\lambda)$
* Two useful properties:

  > If the number of events occuring in a unit of time is a Poisson rv with parameter $\lambda$, then the time between events is exponential, also with parameter $\lambda$.

  > The memoryless property of the exponential rv: If $X \sim Exp(\lambda)$, then $P(X \gt s + t \mid X \gt s) = P(X \gt t)$ for all $s,t \ge 0$.


