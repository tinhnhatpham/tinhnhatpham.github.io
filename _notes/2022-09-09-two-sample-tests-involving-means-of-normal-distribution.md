---
layout: post
comments: false
title: Two-sample Tests Involving Means of Normal Distributions
categories: [t-Tests and Two-Sample Tests]
---

### **Example**

Fifth grade students from two neighboring
counties took a placement exam.

* Group 1, from County 1, consisted of
57 students. The sample mean score
for these students was 77.2 and the
true variance is known to be 15.3.

* Group 2, from County 2, consisted of
63 students and had a sample mean
score of 75.3 and the true variance is
known to be 19.7.

From previous years of data, it is
believed that the scores for both
counties are normally distributed.

<i>Derive a test to determine whether or
not the two population means are the
same.</i>

$$
  H_0:\mu_1=\mu_2\\
  H_1:\mu_1\neq\mu_2
$$

* Suppose that $X_{1,1},X_{1,2},...,X_{1,n_1}$ is a random sample of size $n_1$ from the normal distribution with mean $\mu_1$ and variance $\sigma_1^2$.

* Suppose that $X_{2,1},X_{2,2},...,X_{2,n_2}$ is a random sample of size $n_2$ from the normal distribution with mean $\mu_2$ and variance $\sigma_1^2$.

* Suppose that $\sigma_1^2$ and $\sigma_2^2$ are known and that the samples are independent.

We can re-written the hypotheses:

$$
  H_0:\mu_1=\mu_2\qquad H_1:\mu_1\neq\mu_2\\
  \Downarrow\\
  H_0:\mu_1-\mu_2=0\\
  H_1:\mu_1-\mu_2\neq0
$$

And we can think of this as

$$
  \theta=0\text{ versus }\theta\neq0\\
  \text{for}\\
  \theta=\mu_1-\mu_2
$$

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\theta=\mu_1-\mu_2$.

$$
  \widehat{\theta}=\overline{X}_1-\overline{X}_2
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

Reject $H_0$, in favor of $H_1$ if either

$$
  \widehat{\theta}\gt c\text{ or }\widehat{\theta}\lt-c
$$

for some $c$ to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$ using $\alpha$.

Will be working with the random variable

$$
  \overline{X}_1-\overline{X}_2
$$

We need to know its distribution. Distribution of the statistic:

* $\overline{X}_1-\overline{X}_2$ is normally distributed.

* Mean:

  $$
    \begin{align}
      E[\overline{X}_1-\overline{X}_2]&=E[\overline{X}_1]-E[\overline{X}_2]\\
      &=\mu_1-\mu_2
    \end{align}
  $$

* Variance:

  $$
    \begin{align}
      Var&[\overline{X}_1-\overline{X}_2]=Var[\overline{X}_1+(-1)\overline{X}_2]\\
      &\stackrel{indep}{=}Var[\overline{X}_1]+Var[(-1)\overline{X}_2]\\
      &=Var[\overline{X}_1]+(-1)^2Var[\overline{X}_2]\\
      &=Var[\overline{X}_1]+Var[\overline{X}_2]\\
      &=\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}
    \end{align}
  $$

We know

$$
  \overline{X}_1-\overline{X}_2\sim N\Bigg(\mu_1-\mu_2,\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}\Bigg)\\
  Z=\frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}\sim N(0,1)\\
$$

$$
  \begin{align}
    \alpha&=P(\text{Type I Error})\\
    &=P(\text{Reject }H_0;\theta=0)\\
    &=P(\overline{X}_1-\overline{X}_2\gt c\text{ or }\overline{X}_1-\overline{X}_2\lt-c;\theta=0)\\
    &=1-P(-c\le\overline{X}_1-\overline{X}_2\le c;\theta=0)\\
  \end{align}
$$

* Subtract $\mu_1-\mu_2$ (which is 0).

* Devide by $\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}$

$$
  \alpha=1-P\Bigg(\frac{-c}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}\le Z\le\frac{c}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}\Bigg)\\
  1-\alpha=P\Bigg(\frac{-c}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}\le Z\le\frac{c}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}\Bigg)\\
$$

![png](\assets\images\notes\two-sample-tests-involving-means-of-normal-distribution.png)

$$
  \implies \frac{c}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}=z_{\alpha/2}\\
$$

<font color='blue'><b>Step Four:</b></font>

Conclusion:

Reject $H_0$, in favor of $H_1$, if

$$
  \overline{X}_1-\overline{X}_2\gt z_{\alpha/2}\frac{c}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}
$$

or

$$
  \overline{X}_1-\overline{X}_2\lt -z_{\alpha/2}\frac{c}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}
$$

---

$$
  \begin{align}
    &n_1=57 &&n_2=63\\
    &\overline{X}_1=77.2 &&\overline{x}_2=75.3\\
    &\sigma_1^2=15.3&&\sigma_2^2=19.7
  \end{align}
$$

Suppose that $\alpha=0.05$. In R:

```R
qnorm(0.975)
```

$$
  z_{\alpha/2}=z_{0.025}=1.96\\
  z_{\alpha/2}\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}=1.49\\
  \overline{x}_1-\overline{x}_2=77.2-75.3=1.9
$$

So,

$$
  \overline{x}_1-\overline{x}_2\gt 
  z_{\alpha/2}\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}
$$

and we reject $H_0$. The data suggests that the true mean scores for the counties are different!
