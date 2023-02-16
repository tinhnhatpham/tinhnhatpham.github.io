---
layout: post
comments: false
title: Small Sample Confidence Intervals for the Difference Between Population Means
categories: [Confidence Intervals Involving the Normal Distribution]
---

### Small Sample Confidence Intervals for the Difference Between Population Means

* Suppose that $X_{1,1},X_{1,2},...X_{1, n_1}$ is a random sample size $n_1$ from the normal distribution with mean $\mu_1$ and variance $\sigma_1^2$.
* Suppose that $X_{2,1},X_{2,2},...X_{2, n_2}$ is a random sample size $n_2$ from the normal distribution with mean $\mu_2$ and variance $\sigma_2^2$.
* Suppose that $\sigma_1^2$ and $\sigma_2^2$ are **unknown** and that the samples are independent.
* Suppose that one or both sample sizes are **small**. $(n_1\le30\text{ and/or }n_2\le30)$

Goal: Find a $100(1-\alpha)\%$ confidence interval for $\mu_1-\mu_2$.

*Huge Assumption:* $\sigma_1^2=\sigma_2^2$

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

Because we assumed that $\sigma_1^2=\sigma_2^2$, we have:

$$
  \begin{align}
    \frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma^2}{n_1}+\frac{\sigma^2}{n_2}}}&=\frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{\sigma^2\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)}} \quad\text{(1)}
  \end{align}
$$

We have $S_1^2$ and $S_2^2$ which are two independent estimators of the common variance $\sigma^2$. How can we combine them?

Sample Info:

* Sample of size $n_1$ from $N(\mu_1,\sigma_1^2)$ with $\bar{X}_1$ and $S_1^2$ reported.
* Sample of size $n_2$ from $N(\mu_2,\sigma_2^2)$ with $\bar{X}_2$ and $S_2^2$ reported.

Assumed that $\sigma_1^2=\sigma_2^2$.

Use a weighted average that gives more weight to the one from the larger sample.

**Pooled Variance:**

$$
  S_p^2=\frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2} \quad\text{(2)}
$$

* We know that

  $$
    \frac{(n_1-1)S_1^2}{\sigma^2}\sim\chi^2(n_1-1)
  $$

  and

  $$
    \frac{(n_2-1)S_2^2}{\sigma^2}\sim\chi^2(n_2-1)
  $$

* These two Chi-Squared above are independent, so we have the sum:

  $$
    \frac{(n_1-1)S_1^2}{\sigma^2}+\frac{(n_2-1)S_2^2}{\sigma^2}\sim\chi^2(n_1+n_2-2)
  $$

We multiply both sides of equation (2) to $\frac{(n_1+n_2-2)}{\sigma^2}$

$$
  \begin{align}
    &\frac{(n_1+n_2-2)}{\sigma^2}S_p^2\\
    &= \frac{(n_1-1)S_1^2}{\sigma^2}+\frac{(n_2-1)S_2^2}{\sigma^2}\sim\chi^2(n_1+n_2-2)
  \end{align}
$$

We put the pooled variance (2) to (1)

$$
  \begin{align}
    &\frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{S_p^2\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)}}\\
    &=\frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{\sigma^2\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)}}\Bigg/\sqrt{\frac{S_p^2}{\sigma^2}}\\
    &=\frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{\sigma^2\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)}}\Bigg/\sqrt{\bigg(\frac{(n_1+n_2-2)S_p^2}{\sigma^2}\bigg)\bigg/(n_1+n_2-2)}\\
  \end{align}
$$

We can see the term above is A $N(0,1)$ divided by the square root of a $\chi^2$ divided by its degrees of freedom. We have

$$
  \frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{S_p^2\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)}}\sim t(n_1+n_2-2)
$$

We can solve for the $\mu_1-\mu_1$ "in the middle":

$$
  -t_{\alpha/2,n_1+n_2-2}\lt\frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{S_p^2\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)}}\lt t_{\alpha/2,n_1+n_2-2}
$$

We get

$$
  \bar{X}_1-\bar{X}_2\pm t_{\alpha/2,n_1+n_2-2}\sqrt{S_p^2\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)}
$$

---

### Example

Calculate 90% confidence interval for $\mu_1-\mu_2$.

* $n_1=9$, $\bar{x}_1=23.2$, $s_1^2=4.3$
* $n_2=8$, $\bar{x}_2=24.7$, $s_2^2=5.2$

We calculate the pooled variance

$$
  S_p^2=\frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}=4.72
$$

And the $t$ critical values can be calculated in R: ```qt(0.95,15)```

$$
  t_{0.05,15}=1.753
$$

From 

$$
  \bar{X}_1-\bar{X}_2\pm t_{\alpha/2,n_1+n_2-2}\sqrt{S_p^2\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)}
$$

We plug in all the numbers

$$
  23.2-24.7\pm 1.753\sqrt{4.72\bigg(\frac{1}{9}+\frac{1}{8}\bigg)}
$$

We get the confidence interval: $(-3.351,0.351)$. We can see that the interval actually contains the value $0$. That saying that it is plausible or possible that $\mu_1-\mu_2=0$ or $\mu_1=\mu_2$.

What if we can't say that $\sigma_1^2$ is equal to $\sigma_2^2$?

This is hard and is known as the **Behrens-Fisher problem**. And the most popular solution to this problem is **Welch's Approximation**:

$$
  T=\frac{\bar{X_1}-\bar{X_2}-(\mu_1-\mu_2)}{\sqrt{\frac{S_1^2}{n_1}+\frac{S_2^2}{n_2}}}\stackrel{approx}{\sim}t(\nu)
$$

Where $\nu$ is given by:

$$
  \nu=\frac{\bigg(\frac{S_1^2}{n_1}+\frac{S_2^2}{n_2}\bigg)^2}{\frac{(S_1^2/n_1)^2}{n_1-1}+\frac{(S_2^2/n_2)^2}{n_2-1}}
$$

We can use R to calculate instead calculating by hand:

```R
x<-rnorm(10)
x<-rnorm(14)
t.test(x,y,conf.level=0.90)
```

We will get the result says that

> 90 percent confidence interval: -0.6289463 0.6514495