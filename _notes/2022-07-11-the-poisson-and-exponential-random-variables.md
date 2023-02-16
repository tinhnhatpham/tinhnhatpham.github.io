---
layout: post
comments: false
title: The Poisson and Exponential Random Variables
categories: [Continuous Random Variables]
---

### **The Poisson Random Variable**

The number of customers who arrive for service, and their waiting times, are described by the Poisson rv and the exponential rv, respectively.

A Poisson rv is a discrete rv that describes the total number of events that happen in a certain time period.

* \# of vehicles crossing a bridge in one day.
* \# of gamma rays hitting a satellite per hour.
* \# of cookies sold at a bake sale in one hour.
* \# of customers arriving at a bank in a week.

**Definition:** A discrete random variable $X$ has **Possion distribution** with parameter $\lambda(\lambda\gt0)$ if the probability mass function of $X$ is

$$
  \color{red}{
  P(X=k)=\frac{\lambda^k}{k!}e^{-λ}}\quad\text{for }k=0,1,2,\ldots
$$

Verify:

$$
  \begin{align}
    \sum_{k=0}^{\infty}P(X=k)&=\sum_{k=0}^{\infty}\frac{\lambda^k}{k!}e^{-\lambda}\\
    &=e^{-\lambda}\underbrace{\sum_{k=0}^{\infty}\frac{\lambda^k}{k!}}_{e^\lambda}\\
    &=e^{-\lambda}e^\lambda\\
    &= 1
  \end{align}
$$

**Expected value**

$$
  \begin{align}
    \color{red}{E(X)}&=\sum_{k=0}^{\infty}kP(X=k)\\
    &=\sum_{k=0}^{\infty}k\frac{\lambda^k}{k!}e^{-\lambda}\\
    &=\lambda\underbrace{\sum_{k=1}^{\infty}\frac{\lambda^{k-1}}{(k-1)!}}_{e^\lambda}e^{-\lambda}\\
    &=\color{red}{\lambda}
  \end{align}
$$

**The second moment**

$$
  \begin{align}
    E(X^2)&=\sum_{k=0}^{\infty}k^2P(X=k)\\
    &=\sum_{k=0}^{\infty}k^2\frac{\lambda^k}{k!}e^{-\lambda}\\
    &=\lambda(\lambda+1)
  \end{align}
$$

**Variance**

$$
  \begin{align}
    \color{red}{V(X)}&=E(X^2) - (E(X))^2\\
    &=\lambda(\lambda+1)-\lambda^2\\
    &=\color{red}{\lambda}
  \end{align}
$$

**Notation:** $X\sim\text{Poisson}(\lambda)$

Example: The number of mosquitoes captured in a trap during a given period of time can be modeled by a Poisson with $\lambda=4.5$. What is the probability that the trap contains exactly 5 mosquitoes? 5 of fewer mosquitoes?

$X=\text{# of mosquitoes}$

$X\sim\text{Poisson}(\lambda=4.5)$

$$
  \begin{align}
    P(X=5)&=\frac{(4.5)^5}{5!}e^{-4.5}\\
    &\approx0.1708\\
    P(X\le5)&=\sum_{k=0}^5P(X=k)\\
    &=\sum_{k=0}^5\frac{(4.5)^k}{k!}e^{-4.5}\\
    &\approx0.7029
  \end{align}
$$

Example: A factory makes parts for a medical device company. On average, the rate of defective parts per day is 10. You are responsible for monitoring the number of defective parts on a particular day.

* Define a appropriate random variable for this experiment.
* Give the values that the random variable can take on.
* Find the probability that the random variable equals 2.
* What assumptions do you need to make?

Let $X=\text{# of defective parts (that day)}$

Model $X\sim\text{Poisson}(\lambda=10)$, $X\in{0,1,2,\ldots}$

Assumption: $X$, as a Poisson, can take on an infinite number of values, but we can't make an infinite \# of parts.

$$
  \begin{align}
    P(X=2)&=e^{-\lambda}\frac{\lambda^2}{2!}\\
    &=e^{-10}\frac{(10)^2}{2!}\\
    &\approx0.0023
  \end{align}
$$

---

### **The Exponential Random Variable**

The family of exponential distributions provides probability models that are widely used in engineering and science disciplines to describe **time-to-event data**. An exponential rv is continuous.

* Time until birth.
* Time until a light bulb fails.
* Waiting time in a queue.
* Length of service time.
* Time between customer arrivals.

**Definition:** A continuous random variable $X$ has the exponential distribution with rate parameter $\lambda(\lambda\gt0)$ if the pdf of $X$ is:

$$
  f(x)=\begin{cases}
    \begin{align}
      &\lambda e^{-\lambda x},&&x\ge 0\\
      &0,&&\text{else}
    \end{align}
  \end{cases}
$$

Verify:

$$
  \begin{align}
    \int_{-\infty}^{\infty}f(x)dx&=1\\
    &\Downarrow\\
    \int_{0}^{\infty}\lambda e^{-\lambda x}dx&=\lim_{t \to 0}\int_0^{t}\lambda e^{-\lambda x}dx\\
    &=\lim_{t \to 0}\frac{\lambda}{-\lambda}e^{-\lambda x}\bigg\rvert_{0}^{t}\\
    &=\lim_{t \to 0}(-e^{-\lambda t}+1)\\
    &=1
  \end{align}
$$

**Expected value**

$$
  \begin{align}
    \color{red}{E(X)}&=\int_{}^{}x.\lambda e^{-\lambda x}dx\\
    &=\color{red}{\frac{1}{\lambda}}
  \end{align}
$$

**The second moment**

$$
  \begin{align}
    E(X^2)&=\int_0^{\infty}x^2\lambda e^{-\lambda x}\\
    &=\frac{2}{\lambda^2}
  \end{align}
$$

**Variance**

$$
  \begin{align}
    \color{red}{V(X)}&=E(X^2)-(E(X))^2\\
    &=\frac{2}{\lambda^2}-\bigg(\frac{1}{\lambda}\bigg)^2\\
    &=\color{red}{\frac{1}{\lambda^2}}
  \end{align}
$$

**Notation** $X\sim\text{exp}(\lambda)$

**Useful properties of the exponential**

First, if the number of events occurring in a unit of time is a Poisson rv with parameter $\lambda$, then the time between events is exponential, also with parameter $\lambda$.

Example: Suppose the number of customers arriving for service is modeled by a Poisson rv with $\lambda=5$. That is, an average of 5 customers arrive per hour. Then, the time between arrivals is exponential with $1/\lambda=1/5$. That is, the expected time between arrivals is $1/5$ hour.

The second important property is the memoryless property of the exponential rv: If $X\sim\text{Exp}(\lambda)$, then

$$
  \color{red}{P(X\gt s+t\vert X\gt s)=P(X\gt t)\quad\text{for all }s,t\gt0}
$$

<u>Right hand side:</u>

$$
  \begin{align}
    P(X\gt t)&=1-P(X\le t)\\
    &=1 - \int_0^{t}\lambda e^{-\lambda x}dx\\
    &=1 - \bigg(\frac{\lambda}{-\lambda}e^{-\lambda x}\bigg)\Bigg\rvert_{0}^{t}\\
    &=1-e^{-\lambda t} - 1\\
    &=e^{-\lambda t}
  \end{align}
$$

<u>Left hand side:</u>

$$
  \begin{align}
    P(X\gt s+t\vert X\gt s)&=\frac{P(X\gt s+t\cap X\gt s)}{P(X\gt s)}\\
    &=\frac{P(X\gt s+t)}{P(X\gt s)}\\
    &=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}\\
    &=e^{-\lambda t}
  \end{align}
$$

**Example:** Suppose the service time at a bank with one teller is modeled by a rv with $X\sim\text{Exp}(\lambda=1/5)$. Then, $E(X)=1/\lambda=5$ minutes. If there is a customer in service when you enter the bank, find the probability that the customer is still in service 4 minutes later.

Let $X$ = service time of customer (starting from your entrance into bank)

$$
  P(X\ge4)=\int_4^\infty\lambda e^{-\lambda x}dx=e^{-4/5}\approx0.449
$$

In R:

```R
lambda = 1/5
integrand = function(x){
  lambda*exp(-lambda*x)
}

integrate(integrand, lower=4, upper=Inf)
```

Suppose that when you enter the bank, you know that customer started service 5 minutes ago. Then, what is the probability that they need at least 4 more minutes in service?

$$
  \begin{align}
    P(X\ge9\vert X\ge5)&=\frac{P(X\ge9\cap X\ge5)}{P(X\ge5)}\\
    &=\frac{P(X\ge9)}{P(X\ge5)}\\
    &=\frac{e^{-9/5}}{e^{-1}}\\
    &=e^{-4/5}
  \end{align}
$$