---
layout: post
comments: false
title: Confidence Intervals for The Difference Between Population Means
categories: [Confidence Intervals Involving the Normal Distribution]
---

### Confidence Interval for The Difference Between Population Means
* Suppose that $X_{1,1},X_{1,2},...X_{1, n_1}$ is a random sample size $n_1$ from the normal distribution with mean $\mu_1$ and variance $\sigma_1^2$.

* Suppose that $X_{2,1},X_{2,2},...X_{2, n_2}$ is a random sample size $n_2$ from the normal distribution with mean $\mu_2$ and variance $\sigma_2^2$.

* Suppose that $\sigma_1^2$ and $\sigma_2^2$ are known and that the samples are independent.

Find a $100(1-\alpha)\%$ confidence interval for the difference $\mu_1-\mu_2$.

**Step One:**

An estimator: $\bar{X_1}-\bar{X_2}$

**Step Two:**

Distribution of the estimator:

* $\bar{X_1}\sim N(\mu_1,\sigma_1^2/n_1)$
* $\bar{X_2}\sim N(\mu_2,\sigma_2^2/n_2)$
* $\bar{X_1} - \bar{X_2}$ is normally distributed
* Mean: 

  $$
    \begin{align}
      E[\bar{X_1}-\bar{X_2}] &= E[\bar{X_1}] - E[\bar{X_2}]\\
      &= \mu_1-\mu_2
    \end{align}
  $$

* Variance:

  $$
    \begin{align}
      Var[\bar{X_1}-\bar{X_2}]&=Var[\bar{X_1}+ (-1)\bar{X_2}]\\
      &=Var[\bar{X_1}]+Var[(-1)\bar{X_2}]\\
      &=Var[\bar{X_1}]+(-1)^2Var[\bar{X_2}]\\
      &=Var[\bar{X_1}]+Var[\bar{X_2}]
    \end{align}
  $$

So we know that our difference of sample means is normally distributed

$$
  \bar{X_1}-\bar{X_2}\sim N(\mu_1-\mu_2,\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2})\\
  \Downarrow\\
  Z = \frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}\sim N(0,1)
$$

**Step Three:**

Critical values:

$$
  -z_{\alpha/2}\lt \frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}\lt z_{\alpha/2}
$$

**Step Four:**

Solve for $\mu_1-\mu_2$ "in the middle". We get

$$
  \bar{X_1}-\bar{X_2}\pm z_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}
$$

---

### Example

Fifth grade students from two neighboring counties took a placement exam.

Group 1, from county A, consisted of 57 students. The sample mean score for these these students was 77.2. Group 2, from county B, consisted of 63 students and had a sample mean score of 75.3.

From previous years of data, it is believed that the scores for both counties are normally distributed, and the variances of scores from counties A and B, repectively, are 15.3 and 19.7.

Find and interpret a 99% confedence interval for $\mu_1-\mu_2$, the difference in the means for the counties.

We have

* $n_1=57$, $\bar{x_1}=77.2$, $\sigma_1^2=15.3$
* $n_2=63$, $\bar{x_2}=75.3$, $\sigma_2^2=19.7$

![png](\assets\images\notes\confidence-intervals-for-the-difference-between-population-means-1.png)

We put all the numbers to the equation:

$$
  \bar{X_1}-\bar{X_2}\pm z_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}
$$

The result is: $(-0.0515, 3.8515)$

This is an interval of "plausible values" for the difference $\mu_1-\mu_2$.

Since it contains the values $0$, it is plausible that $\mu_1-\mu_2=0$. It is plausible that $\mu_1=\mu_2$.

---

For large samples $(n_1\gt30\quad n_2\gt30)$

$$
  \bar{X_1}-\bar{X_2}\pm z_{\alpha/2}\sqrt{\frac{S_1^2}{n_1}+\frac{S_2^2}{n_2}}
$$

is an approximate $100(1-\alpha)\%$ confidence interval for $\mu_1-\mu_2$.