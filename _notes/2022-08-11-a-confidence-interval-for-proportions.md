---
layout: post
comments: false
title: A Confidence Interval for Proportions
categories: [Confidence Intervals Beyond the Normal Distribution]
---

### Example

A random sample of 500 people in a
certain country which is about to have a
national election were asked whether they
preferred “Candidate A” or “Candidate B”. 

From this sample, 320 people responded
that they preferred Candidate A.

Construct an approximate 95% confidence
interval for the true proportion of people in
the country who prefer Candidate A. 

Let $p$ be the true proportion of the
people in the country who prefer
Candidate A.

We have an estimate

$$
  \hat{p} = \frac{320}{500}=\frac{16}{25}
$$

The estimator is

$$
  \hat{p}=\frac{\text{# in the sample who like A}}{\text{# in the sample}}
$$

**The model:**

Take a random sample size $n$. Record $X_1,X_2,...,X_n$ where

$$
  {\huge{X_i}} = \begin{cases}
    1 \text{ person i likes Candidate A}\\
    \\
    0 \text{ person i likes Candidate B}
  \end{cases}
$$

Then $X_1,X_2,...,X_n$ is a random sample from the Bernoulli distribution with parameter $p$. Note that, with these 1's and 0's,

$$
  \begin{align}
    \hat{p}&=\frac{\text{# in the sample who like A}}{\text{# in the sample}}\\
    \\
    &=\frac{\sum_{i=1}^{n}X_i}{n}=\bar{X}
  \end{align}
$$

By the Central Limit Theorem, $\hat{p}=\bar{X}$
has, for large samples, an approximately
normal distribution. We know that

$\hat{p}=\bar{X}$

$E[\hat{p}] = E[X_1] = p$

$Var[\hat{p}]=\frac{Var[X_1]}{n}=\frac{p(1-p)}{n}$

So

$$
  \hat{p}\stackrel{approx}{\sim}N\bigg(p,\frac{p(1-p)}{n}\bigg)
$$

In particular, assuming we have a large sample

$
  \frac{\hat{p}-p}{\sqrt{\frac{p(1-p)}{n}}}\text{behaves roughly like a $N(0,1)$ as $n$ gets large}
$

What does "large" mean?

"n>30" is a rule of thumb to apply to all distribution, but we can (and should!) do better with specific distributions.

* $\hat{p}$ lives between 0 and 1.
* The normal distribution lives between $-\infty$ and $\infty$.
* However, 99.7% of the area under a $N(0,1)$ curve lies between -3 and 3.

$$
  \hat{p}\stackrel{approx}{\sim}N\bigg(p,\frac{p(1-p)}{n}\bigg)\\
  \implies \sigma_{\hat{p}}=\sqrt{\frac{p(1-p)}{n}} 
$$

However, this quantity is unknown to us. In practice, we approximate it with the estimator

$$
  \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

Go forward using normality if the interval, which is our estimator minus three standard deviations up to the estimator plus three standard deviations

$$
  \Bigg(\hat{p}-3\sqrt{\frac{\hat{p}(1-\hat{p})}{n}},\hat{p}+3\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\Bigg)
$$

is completely contained within $[0,1]$. 

We can standardize it into a $N(0,1)$ and put it in between two critical values.

$$
  -z_{\alpha/2}\lt \frac{\hat{p}-p}{\sqrt{\frac{p(1-p)}{n}}}\lt z_{\alpha/2}
$$

Although it looks difficult to isolate $p$ "in the middle", it can be done.

However, it is far more common to just plug $\hat{p}$ in for the $p$'s in the denominator to get

$$
  \hat{p}\pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

as an approximate $100(1-\alpha)\%$ confidence interval for $p$.

**Back to the example:**

Let $p$ be the true proportion of the people in the country who prefer Candidate A. Find a 95% confidence interval for $p$.

We have $n=500$, $\hat{p}=\frac{16}{25}$

Now we check whether or not this interval is fully contained in the interval from 0 to 1

$$
  \Bigg(\hat{p}-3\sqrt{\frac{\hat{p}(1-\hat{p})}{n}},\hat{p}+3\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\Bigg)=(0.5756,0.7704)
$$

Because we want a 95% confidence interval and we are talking about standard normal distribution

$95\% \quad \implies z_{0.025}=1.96$ ```qnorm(0.975)=1.96```

$$
  n=500, \quad \hat{p}=\frac{16}{25}\\
  \hat{p}\pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\\
  \Downarrow \\
  (0.5979,0.6821)
$$
