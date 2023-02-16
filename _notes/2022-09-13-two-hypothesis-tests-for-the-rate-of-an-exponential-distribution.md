---
layout: post
comments: false
title: Two Hypothesis Tests for the Rate of an Exponential Distribution
categories: [Hypothesis Testing Beyond Nomality]
---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

Derive a hypothesis test of size $\alpha$ for

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda\gt\lambda_0
$$

What statistic should we use?

### <center><font color='green'><b><u>Test1: Using the Sample Mean</u></b></font></center>

<font color='blue'><b>Step One:</b></font>

Choose a statistic.

$$
  \overline{X}
$$

<font color='blue'><b>Step Two:</b></font>

![png](\assets\images\notes\two-hypothesis-tests-for-the-rate-of-an-exponential-distribution.png)

We have

$$
  f(x)=\lambda e^{-\lambda x}\\
  E[X] = 1/\lambda
$$

* If we have a higher $\lambda$, we will see generally smaller values comming out of the distribution.

* If we have a higher $\lambda$, the random variable has a smaller mean.

* Overall, if we're sampling values from the exponential, they're going to be smaller as $\lambda$ gets larger.

Give the form of the test.

Reject $H_0$, in favor of $H_1$, if

$$
  \overline{X}\lt c
$$

for some $c$ to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha &= P(\text{Type I Error})\\
    &= P(\text{Reject }H_0;\lambda_0)\\
    &= P(\overline{X}\lt c;\lambda_0)\\
    &= P(W\lt2n\lambda_0c)\\
  \end{align}
$$

![png](\assets\images\notes\two-hypothesis-tests-for-the-rate-of-an-exponential-distribution-1.png)

We want

$$
  2n\lambda_0c=\chi^2_{1-\alpha,2n}
$$

<font color='blue'><b>Step Four:</b></font>

Conclusion.

Reject $H_0$, in favor of $H_1$, if

$$
  \overline{X}\lt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
$$

<u>Note:</u>

To find $\chi^2_{\alpha,n}$.

In R, if we want to get $\chi^2_{0.10,6}$.

```R
qchisq(0.90, 6)
```

---

### <center><font color='green'><b><u>Test1: Using the Sample Minimum</u></b></font></center>

<font color='blue'><b>Step One:</b></font>

Choose a statistic.

$$
  Y_n=\text{min}(X_1,X_2,...,X_n)
$$

<font color='blue'><b>Step Two:</b></font>

![png](\assets\images\notes\two-hypothesis-tests-for-the-rate-of-an-exponential-distribution.png)

We have

$$
  f(x)=\lambda e^{-\lambda x}\\
  E[X] = 1/\lambda
$$

* If the alternate hypothesis is true, then $\lambda$ is large, that means that values coming out of the exponential distribution are actually smaller than in the null hypothesis.

Give the form of the test.

Reject $H_0$, in favor of $H_1$, if 

$$
  Y_n\lt c
$$

for some $c$ to be determined.

<font color='blue'><b>Step Three:</b></font>

Find $c$.

$$
  \begin{align}
    \alpha &= P(\text{Type I Error})\\
    &= P(\text{Reject }H_0;\lambda_0)\\
    &= P(Y_n\lt c;\lambda_0)\\
    &= P(n\lambda_0Y_n\lt cn\lambda_0;\lambda_0)\\
    &= P(X\lt cn\lambda_0)\\
  \end{align}
$$

where $X\sim\text{exp}(\text{rate}=1)$.

![png](\assets\images\notes\two-hypothesis-tests-for-the-rate-of-an-exponential-distribution-2.png)

We want the probability that the random variable is less that or equal to some unknown number, and we want that to equal $\alpha$.

$$
  1 - e^{-?}=\alpha\\
  \implies ? = -\ln(1-\alpha)
$$

So

$$
  \alpha=P(X\lt cn\lambda_0)\\
  \Downarrow\\
  cn\lambda_0=-\ln(1-\alpha)\\
  \Downarrow\\
  c=\frac{-\ln(1-\alpha)}{n\lambda_0}
$$

<font color='blue'><b>Step Four:</b></font>

Conclusion.

Reject $H_0$, in favor of $H_1$, if

$$
  Y_n=\text{min}(X_1,X_2,...,X_n)\lt\frac{-\ln(1-\alpha)}{n\lambda_0}
$$

---

### <center><font color='green'><b><u>Compare the Tests</u></b></font></center>

**Test 1:** based on $\overline{X}$

$$
  \begin{align}
    \gamma(\lambda)&=P(\text{Reject }H_0;\lambda)\\
    &=P\Bigg(\overline{X}\lt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0};\lambda\Bigg)\\ 
    &=P\Bigg(2n\lambda\overline{X}\lt2n\lambda\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0};\lambda\Bigg)\\
    &=P\Bigg(W\lt\frac{\lambda}{\lambda_0}\chi^2_{1-\alpha,2n}\Bigg)\\
  \end{align}
$$

So we have the power function for the first test

$$
  \gamma_1{\lambda}=P\Bigg(W\lt\frac{\lambda}{\lambda_0}\chi^2_{1-\alpha,2n}\Bigg)\\
$$

Suppose that

$$
  n=10\quad\lambda_0=1\quad\alpha=0.05\\
  \chi^2_{1-\alpha,2n}=\chi^2_{0.95,20}=10.851
$$

$$
  \begin{align}
    \gamma_1{\lambda}&=P\Bigg(W\lt\frac{\lambda}{\lambda_0}\chi^2_{1-\alpha,2n}\Bigg)\\
    &=P(W\lt10.851\lambda)
  \end{align}
$$

We plot in R:

```R
lambda <- seq(0,6,0.01)
f<-pchisq(10.81*lambda,20)
plot(lambda, f, type='l')
```

![png](\assets\images\notes\two-hypothesis-tests-for-the-rate-of-an-exponential-distribution-3.png)

* For the hypotheses, $\lambda \le \lambda_0$ which $\lambda_0=1$. So we want this power function, the probability we reject to be small when $\lambda < 1$, because if $\lambda$ is less then $1$ the null hypothesis is true and we don't want to reject it when it's true.

* When $lambda \gt \lambda_0$, the region from $1$ and go the the right is where the alternate hypothesis is true, and we want that probability  to be large. 

**Test 2:** based on $Y_n=\text{min}(X_1,X_2,...,X_n)$

$$
  \begin{align}
    \gamma_2(\lambda)&=P(\text{Reject }H_0;\lambda)\\
    &=P\Bigg(\underbrace{Y_n}_{\text{exp}(\text{rate}=n\lambda)}\lt\frac{-(\ln(1-\alpha)}{n\lambda_0};\lambda\Bigg)\\
    &=1 - e^{-n\lambda(-\ln(1-\alpha)/n\lambda_0)}\\
    &=1 - e^{\lambda(-\ln(1-\alpha)/\lambda_0}\\
    &=1 - (1-\alpha)^{\lambda/\lambda_0}\\
  \end{align}
$$

![png](\assets\images\notes\two-hypothesis-tests-for-the-rate-of-an-exponential-distribution-4.png)

* We want the probability that reject the null hypothesis in the region $\lambda\gt\lambda_0$ which $\lambda_0=1$ is larger. 

* We can see in the gragh, the power function of test 1 shows higher probability in region that reject $H_0$, and lower in the region of $H_0$. So the test 1 is much better than the test 2. We should use statistic $\overline{X}$ instead of sample minimum.
