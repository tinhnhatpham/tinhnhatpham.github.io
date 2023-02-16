---
layout: post
comments: false
title: Comparing Population Proportions
categories: [t-Tests and Two-Sample Tests]
---

### **Example**

A random sample of 500 people in a certain **county** which is about to have a national election were asked whether they preferred "Candidate A" or "Candidate B".

From this sample, 320 people responded that they preferred Candidate A.

A random sample of 400 people in a **second county** which is about to have a national election were asked whether they preferred "Candidate A" or "Candidate B".

From this second county sample, 268 people responded that they preferred Candidate A.

$$
  \widehat{p}_1 = \frac{320}{500} = 0.64\\
  \quad\\
  \widehat{p}_2 = \frac{268}{500} = 0.67\\
$$

We want to test that

$$
  H_0:p_1=p_2\qquad H_1:p_1\neq p_2
$$

We can change to

$$
  H_0:p_1-p_2=0\\
  H_1:p_1-p_2\neq0
$$

For large enough samples,

$$
  \widehat{p}_1\stackrel{approx}{\sim}N\bigg(p_1,\frac{p_1(1-p_1)}{n_1}\bigg)
$$

and

$$
  \widehat{p}_2\stackrel{approx}{\sim}N\bigg(p_2,\frac{p_2(1-p_2)}{n_2}\bigg)
$$

We know that $\widehat{p}_1-\widehat{p}_2$ is normally distributed because it's a linear combination of normals.

$$
  \widehat{p}_1-\widehat{p}_2\sim N(?,?)
$$

* Mean:

  $$
    E[\widehat{p}_1-\widehat{p}_2]=E[\widehat{p}_1]-E[\widehat{p}_2]=p_1-p_2
  $$

* Variance:

  $$
    \begin{align}
      Var[\widehat{p}_1-\widehat{p}_2]&\stackrel{indep}{=}Var[\widehat{p}_1]-Var[\widehat{p}_2]\\
      &=\frac{p_1(1-p_1)}{n_1}+\frac{p_2(1-p_2)}{n_2}
    \end{align}
  $$

We have

$$
  \frac{\widehat{p}_1-\widehat{p}_2-(p_1-p_2)}{\sqrt{\frac{p_1(1-p_1)}{n_1}+\frac{p_2(1-p_2)}{n_2}}}\sim N(0,1)\\
$$

Use estimators for $p_1$ and $p_2$ <font color='red'><b>assuming they are the same</b></font>.

* Call the common value $p$.
* Estimate by putting both groups together.

In the example with

$$
  \widehat{p}_1 = \frac{320}{500} = 0.64\quad\widehat{p}_2 = \frac{268}{500} = 0.67\\
$$

We have

$$
  \widehat{p}=\frac{320+268}{500+400}=\frac{588}{900}=\frac{49}{75}\\
  \approx0.6533
$$

Use the fact that:

$$
  \begin{align}
    Z:&=\frac{\widehat{p}_1-\widehat{p}_2-(p_1-p_2)}{\sqrt{\frac{\hat{p}(1-\hat{p})}{n_1}+\frac{\hat{p}(1-\hat{p})}{n_2}}}\stackrel{approx}{\sim} N(0,1)\\
    &=\frac{\widehat{p}_1-\widehat{p}_2-(p_1-p_2)}{\sqrt{\hat{p}(1-\hat{p})\frac{1}{n_1}+\frac{1}{n_2}}}
  \end{align}
$$

This is two-tailed test with z-critical values.

$$
  \begin{align}
    Z&=\frac{0.64-0.67-0}{\sqrt{0.6533(1-0.6533)\bigg(\frac{1}{500}+\frac{1}{500}\bigg)}}\\
    &\approx-0.9397
  \end{align}
$$

Let level of significance $\alpha=0.5$.

![png](\assets\images\notes\comparing-population-proportions.png)

$Z=-0.9397$ does not fall in the rejection region!
