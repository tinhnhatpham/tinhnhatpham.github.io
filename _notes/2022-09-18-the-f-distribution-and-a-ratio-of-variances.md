---
layout: post
comments: false
title: The F-Distribution and a Ratio of Variances
categories: [Hypothesis Testing Beyond Nomality]
---

Suppose that $X_1$ and $X_2$ are independent random variables with

$$
  X_1\sim\chi^2(n_1)\quad\text{and}\quad X_2\sim\chi^2(n_2)\\
$$

Define a new random variable

$$
  F=\frac{X_1/n_1}{X_2/n_2}
$$

$F$ has an "$F$ distribution" with $n_1$ and $n_2$ degrees of freedom.

$$
  F\sim F(n_1,n_2)
$$

![png](\assets\images\notes\the-f-distribution-and-a-ratio-of-variances.png)

PDF:

$$
  f(x;n_1,n_2)=\\
  \frac{1}{B(n_1/2,n_2/2)}\bigg(\frac{n_1}{n_2}\bigg)^{n_1/2}x^{n_1/2}\bigg(1+\frac{n_1}{n_2}x\bigg)^{-(n_1+n_2)/2}\\
  (\text{for }x\gt0)
$$

Mean: $\frac{n_2}{n_2-1}\quad(\text{if }n_2\gt2)$

Variance: $\frac{2n_2^2(n_1+n_2-2)}{n_1(n_2-2)^2(n_2-4)}\quad(\text{if }n_2\gt4)$$

![png](\assets\images\notes\the-f-distribution-and-a-ratio-of-variances-1.png)

In R:

```R
qf(0.95, 5, 1) # 6.608
pf(6.608,5, 1) # 0.9499824
```

The Mean:

$$
  \begin{align}
    E[F]&=E\bigg[\frac{X_1/n_1}{X_2/n_2}\bigg]=\frac{n_2}{n_1}E\bigg[\frac{X_1}{X_2}\bigg]\\
    &\stackrel{indep}{=}\frac{n_2}{n_1}\underbrace{E[X_1]}_{n_1}.E\bigg[\frac{1}{X_2}\bigg]\\
    &=n_2E\bigg[\frac{1}{X_2}\bigg]\\
    &=n_2\int_{-\infty}^{\infty}\frac{1}{x}f_{x_2}(x)dx\\
    &=n_2\int_{0}^{\infty}\frac{1}{x}\frac{1}{\Gamma(n_2/2)}\bigg(\frac{1}{2}\bigg)^{n2/2}x^{n_2/2-1}e^{-x/2}dx\\
    &=n_2\int_{0}^{\infty}\frac{1}{\Gamma(n_2/2)}\bigg(\frac{1}{2}\bigg)^{n2/2}\underbrace{x^{n_2/2-2}e^{-x/2}}_{\text{like a }\Gamma(n_2/2-1,1/2)}dx\\
    &=n_2\frac{\Gamma(n_2/2-1)}{\Gamma(n_2/2)}\frac{1}{2}.\\
    &\qquad\underbrace{\int_{0}^{\infty}\frac{1}{\Gamma(n_2/2-1)}\bigg(\frac{1}{2}\bigg)^{n2/2-1}x^{n_2/2-2}e^{-x/2}dx}_{\text{intergrate to }1}\\
    &=n_2\frac{\Gamma(n_2/2-1)}{(n_2/2-1)\Gamma(n_2/2-1)}\frac{1}{2}\\
    &=\frac{n_2}{n_2-2}
  \end{align}
$$
 
---

* Suppose that $X_{11},X_{12},...,X_{1,n_1}$ is a random sample of size $n_1$ from the $N(\mu_1,\sigma_1^2)$.

* Suppose that $X_{21},X_{22},...,X_{2,n_2}$ is a random sample of size $n_2$ from the $N(\mu_2,\sigma_2^2)$.

Derive a test of size $\alpha$ for

$$
  H_0:\sigma_1^2=\sigma_2^2\quad\text{vs.}\quad H_1:\sigma_1^2\neq\sigma_2^2\\
  \huge{\Downarrow}\\
  H_0:\sigma_1^2/\sigma_2^2=1\quad\text{vs.}\quad H_1:\sigma_1^2/\sigma_2^2\neq1
$$

Let $S_1^2$ and $S_2^2$ be the sample variances for the first and second samples, respectively.

We know that

$$
  \frac{(n_1-1)S_1^2}{\sigma_1^2}\sim\chi^2(n_1-1)
$$

and

$$
  \frac{(n_2-1)S_2^2}{\sigma_2^2}\sim\chi^2(n_2-1)
$$

are independent.

So, define a test statistic $F$ as

$$
  F:=\frac{[(n_1-1)S_1^2/\sigma_1^2]/(n_1-1)}{[(n_2-1)S_2^2/\sigma_2^2]/(n_2-1)}=\frac{\sigma_2^2}{\sigma_1^2}.\frac{S_1^2}{S_2^2}
$$

Then

$$
  \color{red}{F\sim F(n_1-1,n_2-1)}\\
$$

Similarly

$$
  \frac{\sigma_1^2}{\sigma_2^2}.\frac{S_2^2}{S_1^2}\sim F(n_2-1,n_1-1)\\
$$

Under the assumption that $H_0$ is true, we have that

$$
  \frac{S_1^2}{S_2^2}\sim F(n_1-1,n_2-1)
$$

and that

$$
  \frac{S_2^2}{S_1^2}\sim F(n_2-1,n_1-1)
$$

Derive a test of size $\alpha$ for

$$
  H_0:\sigma_1^2=\sigma_2^2\quad\text{vs}\quad H_1:\sigma_1^2\neq\sigma_2^2
$$

* We will reject $H_0$ if $S_1^2/S_2^2$ is too small or too large.

* Equivalently, we reject $H_0$ if $S_2^2/S_1^2$ is too large or too small.

<u>Convention:</u> Put the larger sample variance in the numerator and reject $H_0$ is above the appropriate upper $\alpha/2$ critical value.

---

### **Example**

Fifth grade students from two
neighboring counties took a placement
exam.

Group 1, from County A, consisted of $18$
students. The sample mean score for
these students was $77.2$.

Group 2, from County B, consisted of
$15$ students and had a sample mean
score of $75.3$.

From previous years of data, it is believed
that the scores for both counties are
normally distributed.

The sample variances of scores from Counties A and B, respectively, are $15.3$ and $19.7$.

Derive a test of size $\alpha$ for

$$
  H_0:\sigma_1^2=\sigma_2^2\quad\text{vs}\quad H_1:\sigma_1^2\neq\sigma_2^2
$$

<font color='blue'><b>Step One:</b></font>

Choose a test statistic.

$$
  F:=\frac{S_1^2}{S_2^2}\quad\text{or}\quad F:=\frac{S_2^2}{S_1^2}
$$

Use the one that is greater than 1.

<font color='blue'><b>Step Two:</b></font>

Give the form of the test.

Reject $H_0$, in favor of the alternative if $F$ is either too large or too small.

<font color='green'>Note that this upper tail will have area $\alpha/2$.</font><br>

<font color='blue'><b>Step Three:</b></font>

Find the cutoff using $\alpha/2$.

![png](\assets\images\notes\the-f-distribution-and-a-ratio-of-variances-2.png)

<font color='blue'><b>Step Four:</b></font>

Conclusion.

Reject $H_0$, in favor of $H_1$, if 

$$
  F\gt F_{\alpha/2,n_i-1,n_j-1}
$$

---

$$
  \begin{align}
    &n_1=18\quad s_1^2=15.3\quad \alpha=0.05\\
    &n_2=15\quad s_2^2=19.7
  \end{align}
$$

$$
  F:=\frac{S_2^2}{S_1^2}=\frac{19.7}{15.3}\approx1.288
$$

Critical value:

$$
  F_{0.025,14,17}=2.753
$$

In R:

```R
qf(1-0.025,14,17) #2.753
```

The test statistic $F\approx1.288$ does not fall above

$$
  F_{0.025,14,17}=2.753
$$

Thus we fail to reject $H_0$.