---
layout: post
comments: false
title: P-values and QQ Plots
categories: [Composite Test - Power Functions - and P-Values]
---


### **Hypothesis Testing with P-Values**

### **Example**

In 2019, the average health care annual premium for a family of 4 in the United States, was reported to be $6,015.

In a more recent survey, 100 randomly sampled families of 4 reported an average annual health care premium of $6,537.

Can we say that the true average is  currently greater that $6,015 for all families of 4?

Assume that annual health care premiums are normally distributed with a standard deviation of $814.

Let $\mu$ be the true average for all families of 4.

<font color='blue'><b>Step Zero:</b></font>

Set up the Hypotheses.

$$
  H_0:\mu=6015\qquad H_1:\mu\gt6015
$$

Decide on a level of significance.

$$
  \alpha=0.10
$$

<font color='blue'><b>Step One:</b></font>

Choose an estimator for $\mu$.

$$
  \widehat{\mu}=\overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

Give the "form" of the test.

Reject $H_0$, in favor of $H_1$ if 

$$\overline{X}\gt c$$

for some $c$ to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha&=\underset{\mu=\mu_0}{\text{max}}P(\text{Type I Error};\mu)\\
    &=P(\text{Type I Error};\mu_0)\\
    &=P(\overline{X}\gt c;\mu_0)\\
    &=P\Bigg(\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}\gt\frac{c-6015}{814/\sqrt{100}};\mu_0\Bigg)\\
    &=P\Bigg(Z\gt\frac{c-6015}{814/\sqrt{100}}\Bigg)
  \end{align}
$$

![png](\assets\images\notes\one-tailed-tests-for-the-mean-of-a-normal-distribution-1.png)

We can use ```qnorm(0.90)=1.28``` to calculate the critical value.

$$
  \begin{align}
    &\implies \frac{c-6015}{814/\sqrt{100}}=1.28\\
    &\implies c=6119.19
  \end{align}
$$

<font color='blue'><b>Step Four:</b></font>

Conclusion.

Reject $H_0$, in favor of $H_1$, if 

$$
  \overline{X}\gt6119.19
$$

Our observed sample mean was

$$
  \overline{x}=6537
$$

![png](\assets\images\notes\p-values-and-qq-plots.png)



Our sample mean fell into the rejection region. And that means that if we were to draw a line at out sample mean and shade the area to the right, it would be fully contained in this shaded area like in the picture below

![png](\assets\images\notes\p-values-and-qq-plots-1.png)

* Our sample mean $(6537)$ fell into the <font color='red'>rejection region</font>, so we reject $H_0$.

* Note that the area to the right of our sample mean of $6537$ must be less than $0.10$.

* The area to the right is known as <font color='red'><b>P-value</b></font>.

The $0.10$ is a probability that is relevant when $H_0$ is true.

* This is the $N(6015,814^2/100)$ pdf.

* The red area is $P(\overline{X}\gt6537)$, which is the <font color='red'><b>P-value</b></font> for this problem.

  $$
    \begin{align}
      &P(\overline{X}\gt6537)\\
      &=P\Bigg(\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}\gt\frac{6537-6015}{814/\sqrt{100}}\Bigg)\\
      &=P(Z\gt6.4127)\\
      &\approx0.00000001\\
      &\text{(Super small and way out "in the tail")}
    \end{align}
  $$

So the red are is actually way out there and very, very small. And that means it's in our rejection region, which has size $0.1$. The most common levels of significance for hypothesis tests are $0.05, 0.01$ and $0.1$, but this is so small that it's going to be smaller than any reasonable alpha. And that means we are definitely in the rejection region.

The <font color='red'><b>P-value</b></font> is the area to the right (in this case) of the test statistic $\overline{X}$.

* The <font color='red'><b>P-value</b></font> being less than $0.10$ puts $\overline{X}$ in the rejection region.

* The <font color='red'><b>P-value</b></font> is also less than $0.05$ and $0.01$.

* It looks like we will reject $H_0$ for the most typical values of $\alpha$.

$$
  \text{small P-value} \implies \text{reject } H_0
$$

### **QQ Plots (Quantile Quantile Plots)**

<u>Check for Normality</u>

Quantiles are numbers that divide up the area under a PDF in to equal parts.

Consider the following "data":

$$
  1.678, 2.024, 2.168, 3.018, 1.689,\\
  1.727, 1.743, 3.234, 2.008, 1.309
$$

Now in order:

$$
  1.309, 1.678, 1.689, 1.727, 1.743,\\
  2.008, 2.024, 2.168, 3.018, 3.234
$$

* Note that $10\%$ of the data are at or below $1.309$.
* $20\%$ of the data are ator below $1.678$.

Numbers that divide data into equal groups like this are called <font color='red'>quantiles</font>.

These numbers taken from the sample are called <font color='red'>sample quantiles</font>.

We compare them with numbers that divide the area under the normal curve into 10 equal parts.

![png](\assets\images\notes\p-values-and-qq-plots-2.png)

$$
  \implies \Phi(a)=0.55\\
  \implies a=\Phi^{-1}(0.55)
$$

So $a$ is $\Phi$ inverse of $0.55$, and we can use R to compute:

```R
  qnorm(0.55) = 0.1257
```

Let's have R compare all the quantiles for us.



```R
mysample <- c(1.678, 2.024, 2.168, 3.018, 1.689, 1.727, 1.743, 3.234, 2.008, 1.309)

# Get the Q-Q plot
qqnorm(mysample)

# To estimate a line that minimizes the total sum 
# of the squared error of the distance away from the line.
qqline(mysample)
```


![png](\assets\images\notes\p-values-and-qq-plots-3.png)

