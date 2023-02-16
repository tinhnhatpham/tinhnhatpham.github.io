---
layout: post
comments: false
title: A Review of Maximum Likelihood Estimation
categories: [Likelihood Ratio Tests and Chi-Squared Tests]
---

Flip a possibly unfair coin $n$ times.

Let $p$ be the probability of gettings "Heads" on any one flip. $(0\le p\le1)$

Record $1$'s and $0$'s for H's and T's, respectively.

We now have a random sample

$$
  X_1,X_2,\ldots,X_n\stackrel{iid}{\sim}\text{Bernoulli}(p)
$$

$p$ is unknown and we want to estimate it.

* If the observed data has a lot of $1$'s in it, a higher value of $p$, closer to $1$ is more likely.
* If the observed data has a lot of $0$'s in it, a lower value of $p$, closer to $0$ is more likely.
* If the obsered data has roughly an equal number $0$'s and $1$'s, a value of $p$ closer to $0.5$ is more likely.

The Bernoulli probability mass function is

$$
  f(x;p)=p^x(1-p)^{1-x}
$$

for $x=0,1$.It is zero otherwise.

The joint pmf for $X_1,X_2,\ldots,X_n$ is

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{x};p)&\stackrel{iid}{=}\prod_{i=1}^{n}f(x_i;p)\\
    &=p^{\sum_{i=1}^{n}x_i}(1-p)^{n-\sum_{i=1}^{n}x_i}
  \end{align}
$$

for $x_i\in\{0,1\}$.

$$
  f(\stackrel{\rightharpoonup}{x};p)=\underbrace{P(X_1=x_1,\ldots,X_n=x_n)}_{\text{This is a function of $p$.}}
$$

Find the value of $p$ in $[0,1]$ that makes the probability of seeing $X_1=x_1,X_2=x_2,\ldots,X_n=x_n$ <font color='red'><b>"most likely"</b></font>.

i.e. This is called the <font color='red'><b>maximum likelihood estimator</b></font> for $p$.

$$
   f(\stackrel{\rightharpoonup}{x};p)=p^{\sum_{i=1}^{n}x_i}(1-p)^{n-\sum_{i=1}^{n}x_i}
$$

Think about this as a function of $p$:

$$
  L(p)=p^{\sum_{i=1}^{n}x_i}(1-p)^{n-\sum_{i=1}^{n}x_i}
$$

This is called a <font color='red'><b>likelihood function</b></font>.

It is easier to maximize the <font color='red'><b>log-likelihood</b></font>.

$$
  \begin{align}
    l(p)&=\ln L(p)\\
    &=\left(\sum_{i=1}^{n}x_i\right)\ln p + \left(n-\sum_{i=1}^{n}x_i\right)\ln(1-p)
  \end{align}
$$

This is a function of $p$, to maximize it, we take the derivative w.r.t to $p$ and set it equal to $0$.

$$
  \frac{d}{dp}l(p) = 0\\
  \huge{\Downarrow}\\
  \left(\sum_{i=1}^{n}x_i\right)\frac{1}{p} - \left(n-\sum_{i=1}^{n}x_i\right)\frac{1}{1-p} = 0
$$

The MLE for $p$ is

$$
  \widehat{p}=\frac{\sum_{i=1}^{n}X_i}{n}=\overline{X}
$$

---

For continous $X_1,X_2,\ldots,X_n$, the joint pdf does not represent probability but the MLE is found the same way.

<u>Example</u>

Suppose that $X_1,X_2,\ldots,X_n$ is a random sample from the continous Pareto distribution with pdf

$$
  f(x,\gamma)=\begin{cases}
    \begin{align}
      &\frac{\gamma}{(1+x)^{\gamma+1}}&&x\gt0\\
      &0&&\text{otherwise}\\
    \end{align}
  \end{cases}
$$

The joint pdf is

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{x};\gamma)&\stackrel{iid}{=}\prod_{i=1}^{n}f(x_i;\gamma)\\
    &=\prod_{i=1}^{n}\frac{\gamma}{(1+x)^{\gamma+1}}\\
    &=\frac{\gamma}{\prod_{i=1}^{n}(1+x)^{\gamma+1}}\\
    &=\frac{\gamma}{\left[\prod_{i=1}^{n}(1+x)\right]^{\gamma+1}}
  \end{align}
$$

Equivalently,

$$
  l(\gamma)=n\ln\gamma-(\gamma+1)\sum_{i=1}^{n}\ln(1+x_i)\\
$$

Take derivative w.r.t $\gamma$ and set it equal to $0$

$$
  l'(\gamma)=\frac{n}{\gamma}-\sum_{i=1}^{n}\ln(1+x_i)\stackrel{\text{set}}{=}0\\
$$

The MLE for $\gamma$ is:

$$
  \widehat{\gamma}=\frac{n}{\sum_{i=1}^{n}\ln(1+X_i)}
$$
