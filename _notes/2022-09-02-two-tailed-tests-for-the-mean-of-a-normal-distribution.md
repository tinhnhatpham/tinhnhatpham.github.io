---
layout: post
comments: false
title: Two-Tailed Tests for the Mean of a Normal Distribution
categories: [Composite Test - Power Functions - and P-Values]
---


Let $X_1,X_2,...,X_n$ be a random sample from the normal distribution with mean $\mu$ and <u>known</u> variance $\sigma^2$.

Derive a hypothesis test of size $\alpha$ for testing

$$
  H_0:\mu=\mu_0\\
  H_1:\mu\neq\mu_0
$$

We will look at the sampe mean $\overline{X}$ and reject if it is either too high or too low.

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

<s>Reject $H_0$, in favor of $H_1$ if either $\overline{X}\lt c$ or $\overline{X}\gt d$ for some $c$ and $d$ to be determined.</s>


Easier to make it symmetric!

Reject $H_0$, in favor of $H_1$ if either

<font color='red'>
$$
  \overline{X}\gt\mu_0 + c\\
  \text{or}\\
  \overline{X}\lt\mu_0 - c
$$
</font>
for some c to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha&=\underset{\mu=\mu_0}{\text{max}}P(\text{Type I Error})\\
    &=\underset{\mu=\mu_0}{\text{max}}P(\text{Reject $H_0;\mu$})\\
    &=P(\text{Reject $H_0;\mu_0$})\\
    &=P(\overline{X}\lt\mu_0-c\text{ or }\overline{X}\gt\mu_0+c;\mu_0)\\
    &=1 - P(\mu_0-c\le\overline{X}\le\mu_0+c;\mu_0)\\
  \end{align}
$$

Subtract $\mu_0$ and divide by $\sigma/\sqrt{n}$.

$$
  \begin{align}
    \alpha&=1 - P\Bigg(\frac{-c}{\sigma/\sqrt{n}}\le Z\le\frac{c}{\sigma/\sqrt{n}}\Bigg)\\
    \implies 1-\alpha&=P\Bigg(\frac{-c}{\sigma/\sqrt{n}}\le Z\le\frac{c}{\sigma/\sqrt{n}}\Bigg)\\
  \end{align}
$$

![png](\assets\images\notes\two-tailed-tests-for-the-mean-of-a-normal-distribution.png)

<font color='blue'><b>Step Four:</b></font>

Conclusion:

Reject $H_0$, in favor of $H_1$, if

$$
  \overline{X}\gt\mu_0+z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\\
  \text{or}\\
  \overline{X}\lt\mu_0-z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\\
$$

---

### Example

In 2019, the average health care annual premium for a family of 4 in the United States, was reported to be $6,015.

In a more recent survey, 100 randomly sampled families of 4 reported an average annual health care premium of $6,177.

Can we say that the true average, for all families of 4, is currently different than the sample mean from 2019?

Assume that annual health care premiums are normally distributed with a  standard deviation of $814.

Let $\mu$ be the true average for all families of 4.

Hypotheses:

$$
  H_0:\mu=6015\\
  H_1:\mu\neq6015
$$

We have

$$
  \overline{x}=6177\quad\sigma=814\quad n=100\\
  z_{\alpha/2}=z_{0.025}=1.96
$$

In R: <font color='red'>qnorm(0.975)</font>

$$
  6015+1.96\frac{814}{\sqrt{100}}=6174.5\\
  6015-1.96\frac{814}{\sqrt{100}}=5855.5
$$

![png](\assets\images\notes\two-tailed-tests-for-the-mean-of-a-normal-distribution-1.png)

We reject $H_0$, in favor of $H_1$. The data suggests that the true current average, for all families of 4, is different than it was in 2019.

<u>P-value:</u>

$$
  P(\overline{X}\gt6174.5\text{ or }\overline{X}\lt5855;\mu_0)
$$

![png](\assets\images\notes\two-tailed-tests-for-the-mean-of-a-normal-distribution-2.png)

$$
  \begin{align}
    \text{P-value}&=2P(\overline{X}\gt6177;\mu_0=6015)\\
    &=2P(Z>1.99)\\
    &=2(0.023295) = 0.0466
  \end{align}
$$

```R
# critical value
1-pnorm((6177-6015)/(814/sqrt(100)))
```

This is smaller than $0.05$ so we reject $H_0$ at $0.05$ level of significance.
