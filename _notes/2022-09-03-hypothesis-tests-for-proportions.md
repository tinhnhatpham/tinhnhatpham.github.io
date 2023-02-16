---
layout: post
comments: false
title: Hypothesis Tests for Proportions
categories: [Composite Test - Power Functions - and P-Values]
---


### **Example**

A random sample of $500$ people in a
certain country which is about to have a
national election were asked whether they
preferred “Candidate A” or “Candidate B”.

From this sample, $320$ people responded
that they preferred Candidate A.

Let $p$ be the true proportion of the
people in the country who prefer
Candidate A.

Test the hypotheses

$$
  H_0:p\le0.65\\
  \text{versus}\\
  H_1:p\gt0.65
$$

We have an estimate

$$
  \begin{align}
    \widehat{p}&=\frac{320}{500}=\frac{16}{25}\\
    &=0.64
  \end{align}
$$

<u>The Model:</u>

Take a random sample size $n$.

Record $X_1,X_2,...,X_n$ where

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
  \frac{\widehat{p}-p}{\sqrt{\frac{p(1-p)}{n}}}\text{ behaves roughly like a $N(0,1)$ as $n$ gets large}
$

What does "large" mean?

"n>30" is a rule of thumb to apply to all distribution, but we can (and should!) do better with specific distributions.

* $\hat{p}$ lives between 0 and 1.
* The normal distribution lives between $-\infty$ and $\infty$.
* However, 99.7% of the area under a $N(0,1)$ curve lies between -3 and 3.

$$
  \widehat{p}\stackrel{approx}{\sim}N\bigg(p,\frac{p(1-p)}{n}\bigg)\\
  \implies \sigma_{\widehat{p}}=\sqrt{\frac{p(1-p)}{n}} 
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

<font color='blue'><b>Step One:</b></font>

Choose a statistic.

$$
  \widehat{p}=\text{sample proportion for Candidate A}
$$

<font color='blue'><b>Step Two:</b></font>

Form of the test.

Reject $H_0$, in favor of $H_1$, if $\widehat{p}\gt c$

<font color='blue'><b>Step Three:</b></font>

Use $\alpha$ to find $c$.

Assume normality of $\widehat{p}$?

* It is a sample mean and $n\gt30$.

* The interval

  $$
    \Bigg(\hat{p}-3\sqrt{\frac{\hat{p}(1-\hat{p})}{n}},\hat{p}+3\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\Bigg)=(0.5756,0.7044)
  $$

We have

$$
  \begin{align}
    \alpha&=\underset{p\in H_0}{\text{max}}P(\text{Type I Error})\\
    &=\underset{p\le0.65}{\text{max}}P(\text{Reject }H_0;p)\\
    &=\underset{p\le0.65}{\text{max}}P(\widehat{p}\gt c;p)\\
    &=\underset{p\le0.65}{\text{max}}P\Bigg(\frac{\widehat{p}-p}{\sqrt{\frac{p(1-p)}{n}}}\gt\frac{c-p}{\sqrt{\frac{p(1-p)}{n}}},p\Bigg)\\
    &\approx\underset{p\le0.65}{\text{max}}P\Bigg(Z\gt\frac{c-p}{\sqrt{\frac{p(1-p)}{n}}}\Bigg)
  \end{align}
$$

We're going to graph $\frac{c-p}{\sqrt{\frac{p(1-p)}{n}}}$ as a function of $p$. But we don't have a $c$, so what $c$'s makes sense here?

* Our rejection rule is to reject if $\hat{p}\gt c$. We know that $\hat{p}$ is a sample proportion. So it's not going to make sense to have $c\gt1$, because we will never reject the null hypothesis. So we consider $c$'s between $0$ and $1$ and graph it

![png](\assets\images\notes\hypothesis-tests-for-proportions.png)

The bottom one has $c=0.1$, and the next one up has $c=0.2$, on up the top one has $c=0.9$. We can see these are all decreasing functions of $p$.

We have

$$
  0.10=\underset{p\le0.65}{\text{max}}P\Bigg(Z\gt\frac{c-p}{\sqrt{\frac{p(1-p)}{n}}}\Bigg)
$$

So the probability that a standard normal is greater that a number, and than greater than a lower number. That probability is going up. So if we want to maximize it over all the $p$'s between $0$ and $0.65$, we just plug in the $0.65$, the right end point of that interval.

$$
  \begin{align}
    0.10&=\underset{p\le0.65}{\text{max}}P\Bigg(Z\gt\frac{c-p}{\sqrt{\frac{p(1-p)}{n}}}\Bigg)\\
    &=P\Bigg(Z\gt\frac{c-0.65}{\sqrt{\frac{0.65(1-0.65)}{n}}}\Bigg)\\
    &\implies \frac{c-0.65}{\sqrt{\frac{0.65(1-0.65)}{n}}}=z_{0.10}
  \end{align}
$$

<font color='blue'><b>Step Four:</b></font>

Conclusion:

Reject $H_0$ if

$$
  \widehat{p}\gt0.65+z_{0.10}\sqrt{\frac{0.65(1-0.65)}{n}}
$$

---

<u>Back to the example</u>

Let $p$ be the true proportion of the people in the country who prefer Candidate A.

$$
  n=500\qquad\widehat{p}=\frac{16}{25}=0.64\\
  \alpha=0.10\qquad z_{0.10}=1.28
$$

We have

$$
  \widehat{p}\gt0.65+z_{0.10}\sqrt{\frac{0.65(1-0.65)}{n}}=0.6773
$$

So $\hat{p} = 0.64 \lt0.6773$. We fail to reject $H_0$, in favor of $H_1$.

The data do not suggest that the true proportion of people who like Candidate A is greate than 0.65.
