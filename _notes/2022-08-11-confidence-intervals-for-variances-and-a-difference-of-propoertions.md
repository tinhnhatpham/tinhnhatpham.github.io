---
layout: post
comments: false
title: Confidence Intervals for Variances and a Difference of Proportions
categories: [Confidence Intervals Beyond the Normal Distribution]
---

### Example

A random sample of 500 people in a certain
region of a country, which is about to have
a national election, were asked whether
they preferred "Candidate A" or "Candidate
B".

From this sample, 320 people responded
that they preferred Candidate A.

An independent random sample of 420
people in a **second region** of that country
were asked whether they preferred
"Candidate A" or "Candidate B". 

From this sample, 302 people responded
that they preferred Candidate A.

Construct an approximate 90% confidence
interval for the difference between the true
proportions for each country. 

Let $p_2$ and $p_2$ be the true proportions
for the first and second regions.

We have

$$
  \hat{p}_1=\frac{320}{500}=\frac{16}{25}\text{ and }\hat{p}_2=\frac{302}{420}=\frac{151}{210}\\
  (n_1=500, n_2=420)
$$

Both of these satisfy

$$
  \Bigg(\hat{p}-3\sqrt{\frac{\hat{p}(1-\hat{p})}{n}},\hat{p}+3\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\Bigg)\subseteq [0,1]
$$

**Step One:**

An estimator of $p_1-p_2$ is

$$
  \hat{p}_1 - \hat{p}_2
$$

**Step Two:** By the CLT, we have

$$
  \hat{p}_1\stackrel{approx}{\sim}N\Big(p_1,p_1(1-p_1)/n_1\Big)\\
  \hat{p}_2\stackrel{approx}{\sim}N\Big(p_2,p_2(1-p_2)/n_2\Big)
$$

Find a function of the estimators and the "target" whose distribution is known and "unknown parameter free".

For simplicity and due to the fact that we have large sample sizes, we replace each

$$
  \sigma_{\hat{p}_i}=\frac{p_i(1-p_i)}{n_i}
$$

with the estimator

$$
  \hat{\sigma_{\hat{p}_i}}=\frac{\hat{p}_i(1-\hat{p}_i)}{n_i}
$$

We have 

$$
  \hat{p}_1 - \hat{p}_2\stackrel{approx}{\sim}\\
  N\bigg(p_1-p_2,\frac{\hat{p}_1(1-\hat{p}_1)}{n_1}+\frac{\hat{p}_2(1-\hat{p}_2)}{n_2}\bigg)
$$

Standardize:

$$
  \frac{p_1-p_2-(\hat{p}_1 - \hat{p}_2)}{\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1}+\frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}}\stackrel{approx}{\sim}N(0,1)
$$

**Step Three:**

Put this between appropriate critical values and solve for $p_1-p_2$ "in the middle".

$$
  z_{\alpha/2}=z_{0.10/2}=z_{0.05}=1.645
$$

We have the confidence interval

$$
  \hat{p}_1-\hat{p}_2\pm z_{\alpha/2}\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1}+\frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}
$$

The answer is: $(-0.12953, -0.02857)$

The data suggests that the proportion of people from region 2 that voted for candidate A is larger than the proportion from region 1.

$$
  \hat{p}_1=0.64 \text{ and }\hat{p}_2\approx0.72
$$

---

### Example 2

A potato chip manufacturer sells 10
ounce bags of potato chips. The
company always overfills the bags slightly
so as not to have angry customers.

In addition to overfilling the bags, the
manufacturer wants to make sure that the
standard deviation of weights is small, so
that, even bags on the low fill end will
contain at least the amount of product
advertised.

For a random sample of 20 bags of
chips, the quality control manager
finds that the sample variance is 0.52
ounces.

Assuming that fill weights are normally
distributed, find a 95% confidence
interval for $\sigma$, the true standard
deviation for all bags.

**Step One:**

Decide on an estimator.

We will use the sample standard deviation $S=\sqrt{S^2}$

**Step Two:**

Look at the distribution of $S$. Recall that, for a normal sample,

$$
  \implies \frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)
$$

**Step Three:**

Put the statistic between appropriate critical values and solve for $\sigma$ in the middle.

Here are 3 possibilities. (There are more!)

$$
  \chi^2_{0.975,n-1}\lt\frac{(n-1)S^2}{\sigma^2}\lt\chi^2_{0.025,n-1}\quad\text{(1)}\\
  0\lt\frac{(n-1)S^2}{\sigma^2}\lt\chi^2_{0.05,n-1}\quad\text{(2)}\\
  \chi^2_{0.95,n-1}\lt\frac{(n-1)S^2}{\sigma^2}\lt\infty\quad\text{(3)}
$$

From (1): We can calculate critical values by using R

```
  qchisq(0.025,19) = 8.90652
  qchisq(0.975,19) = 32.85233
```

$$
  8.90652 \lt\frac{(20-1)(0.52)}{\sigma^2}\lt 32.85233\\
  \Downarrow\\
  0.29496\lt\sigma^2\lt 1.08797\\
  \Downarrow\\
  0.54310\lt\sigma\lt1.04306
$$

From (2):

```
  qchisq(0.95,19)=30.14353
```

$$
  0\lt\frac{(20-1)(0.52)}{\sigma^2}\lt30.14353\\
  \Downarrow\\
  0.56698\lt\sigma\lt\infty
$$

From (3): we have

$$
  10.11701\lt\frac{(19-1)(0.52)}{\sigma^2}\lt\infty\\
  \Downarrow\\
  0\lt\sigma\lt0.97867
$$

