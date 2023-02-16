---
layout: post
comments: false
title: Two-Sample t-Tests for a Difference in Two Population Means
categories: [t-Tests and Two-Sample Tests]
---

### **Example**

Fifth grade students from two neighboring
counties took a placement exam.

* Group 1, from County 1, consisted of
8 students. The sample mean score
for these students was 77.2 and the
sample variance is known to be 15.3.

* Group 2, from County 2, consisted of
10 students and had a sample mean
score of 75.3 and the sample variance is
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

* Estimate $\mu_1-\mu_2$ with $\overline{X}_1-\overline{X}_2$.

* $\overline{X}_1-\overline{X}_2$ is normally distributed.

$$
  \frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}\sim ?\\
$$

If at least one sample size is small, the sample variances are not great approximations for the true variances.

* Suppose that $X_{1,1},X_{1,2},...,X_{1,n_1}$ is a random sample of size $n_1$ from the normal distribution with mean $\mu_1$ and variance $\sigma_1^2$.

* Suppose that $X_{2,1},X_{2,2},...,X_{2,n_2}$ is a random sample of size $n_2$ from the normal distribution with mean $\mu_2$ and variance $\sigma_1^2$.

* Suppose that $\sigma_1^2$ and $\sigma_2^2$ are <font color='red'><b>unknown</b></font> and that the samples are independent.

* Suppose that $\sigma_1^2$ and $\sigma_2^2$ are <font color='red'><b>equal</b></font>.

* Since we are assuming that $\sigma_1^2$ = $\sigma_2^2$, there is no need for subscripts.

* Call the common value $\sigma^2$.

* We have two sample variances, $S_1^2$ and $S_2^2$ that we would like to combine into a single estimator for $\sigma^2$.

$$
  S_p^2=\frac{S_1^2+S_2^2}{2}\quad\huge?
$$

We <u>won't</u> use this because:

* If one sample variance is from a larger sample, we'd like to give it more weight.

* We don't know its distribution.

<u>Pooled Variance</u>

Define

$$
  S_p^2=\frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}
$$

Note that

$$
  \frac{(n_1+n_2-2)S_p^2}{\sigma^2}=\underbrace{\frac{(n_1-1)S_1^2}{\sigma^2}}_{\chi^2(n_1-1)}+\underbrace{\frac{(n_2-1)S_2^2}{\sigma^2}}_{\chi^2(n_2-1)}
$$

Because these are independent, we have:

$$
  \frac{(n_1+n_2-2)S_p^2}{\sigma^2}\sim\chi^2(n_1+n_2-2)
$$

We have

$$
  \frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma^2_1}{n_1}+\frac{\sigma^2_2}{n_2}}}\sim N(0,1)\\
  \Downarrow\\
  \frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)\sigma^2}}\\
  \Downarrow\\
  \frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)S_p^2}}\\
$$

We know the distribution if the pooled variance is the true variant. So let put the true variant in

$$
  \begin{align}
    &\frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)S_p^2}}\\
    &=\frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)\sigma^2}}.\sqrt{\frac{\sigma^2}{S_p^2}}\\
    &=\frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)\sigma^2}}\Bigg/\sqrt{\frac{S_p^2}{\sigma^2}}\\
    &=\frac{\frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)\sigma^2}}\quad\sim N(0,1)}{\sqrt{\underbrace{\frac{(n_1+n_2-2)S_p^2}{\sigma^2}}_{\chi^2(n_1+n_2-2)}\bigg/(n_1+n_2-2)}}
  \end{align}
$$

So,

$$
  \frac{\overline{X}_1-\overline{X}_2-(\mu_1-\mu_2)}{\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)S_p^2}}\sim t(n_1+n_2-2)
$$

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\theta=\mu_1-\mu_2$

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

$$
  \begin{align}
    \alpha&=P(\text{Type I Error})\\
    &=P(\text{Reject }H_0;\theta=0)\\
    &=P(\overline{X}_1-\overline{X}_2\gt c\text{ or }\overline{X}_1-\overline{X}_2\lt-c;\theta=0)\\
    &=1-P(-c\le\overline{X}_1-\overline{X}_2\le c;\theta=0)\\
  \end{align}
$$

* Subtract $\mu_1-\mu_2$ (which is $0$).

* Divide by $\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}S_p^2\bigg)}$

$$
  \alpha=1-P(-d\le T\le d)\\
  \text{where}\\
  T\sim t(n_1+n_2-2)\\
  \text{and}\\
  d=c\bigg/\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}S_p^2\bigg)}
$$

We have

$$
  \begin{align}
    &P(-d\le T\le d)=1-\alpha\\
    &\implies d=t_{\alpha/2,n_1+n_2-2}\\
    &\implies c=t_{\alpha/2,n_1+n_2-2}\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}S_p^2\bigg)}
  \end{align}
$$
<font color='blue'><b>Step Four:</b></font>

Conclusion:

Reject $H_0$, in favor of $H_1$, if

$$
  \overline{X}_1-\overline{X}_2\gt t_{\alpha/2,n_1+n_2-2}\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}S_p^2\bigg)}
$$

or

$$
  \overline{X}_1-\overline{X}_2\lt -t_{\alpha/2,n_1+n_2-2}\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}S_p^2\bigg)}
$$

---

**Back to the example**

Fifth grade students from two neighboring
counties took a placement exam.

* Group 1, from County 1, consisted of
8 students. The sample mean score
for these students was 77.2 and the
sample variance is known to be 15.3.

* Group 2, from County 2, consisted of
10 students and had a sample mean
score of 75.3 and the sample variance is
known to be 19.7.

From previous years of data, it is
believed that the scores for both
counties are normally distributed.

Can we say that the true means for Counties A and B are different?

Test the relevant hypotheses at level $0.01$.

$$
  H_0:\mu_1=\mu_2\qquad H_1:\mu_1\neq\mu_2
$$

$$
  \begin{align}
    &n_1=8&&n_2=10\\
    &\overline{x}_1=77.2&&\overline{x}_2=75.3\\
    &S_1^2=15.3&&S_2^2=19.7\\
    &\alpha=0.01&&t_{0.005,16}=2.92
  \end{align}
$$

For the pooled variance:

$$
  \begin{align}
    S_p^2&=\frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}\\
    &=17.775
  \end{align}
$$

For the $c$:

$$
  \begin{align}
    &t_{\alpha/2,n_1+n_2-2}\sqrt{\bigg(\frac{1}{n_1}+\frac{1}{n_2}\bigg)S_p^2}\\
    &=2.92\sqrt{\bigg(\frac{1}{8}+\frac{1}{10}\bigg)(17.77)}\\
    &=5.840\\
  \end{align}
$$

Since $\overline{x}_1-\overline{x}_2=1.9$ is not

* Above 5.840, or
* Below -5.840

We fail to reject $H_0$, in favor of $H_1$ at $0.01$ level of significance.

The data is not indicate there is a significant difference between the true mean scores for counties A and B.