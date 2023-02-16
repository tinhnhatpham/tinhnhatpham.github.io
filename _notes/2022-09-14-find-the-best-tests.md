---
layout: post
comments: false
title: The Neyman-Pearson Lemma - The Best Test
categories: [Hypothesis Testing Beyond Nomality]
---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

Derive a hypothesis test of size $\alpha$ for

$$
  H_0: \lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda=\lambda_1
$$

where $\lambda_1\gt\lambda_0$.

What statistic should we use?

<u>One test has rejection rule:</u>

$$
  \overline{X}\lt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
$$

"Denote" this by $\color{#33D5FF}{\blacksquare}$

$$
  \begin{align}
    \alpha&=P\Bigg(\overline{X}\lt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0};\lambda_0\Bigg)\\
    &=P\Bigg(\color{#33D5FF}{\blacksquare};\lambda_0\Bigg)\\
  \end{align}
$$

<u>One another has rejection rule:</u>

$$
  \text{min}(X_1,X_2,...,X_n)\lt\frac{-\ln(1-\alpha)}{n\lambda_0}
$$

"Denote" this by $\color{orange}\blacksquare$

$$
  \begin{align}
    \alpha&=P\Bigg(\text{min}(X_1,X_2,...,X_n)\lt\frac{-\ln(1-\alpha)}{n\lambda_0};\lambda_0\Bigg)\\
    &=P\Bigg(\color{orange}\blacksquare;\lambda_0\Bigg)\\
  \end{align}
$$

<u>So for all the tests:</u>

$$
  P(\color{#33D5FF}{\blacksquare};\lambda_0)=\alpha\\
  P(\color{orange}{\blacksquare};\lambda_0)=\alpha\\
  P(\color{green}{\blacksquare};\lambda_0)=\alpha\\
  P(\color{yellow}{\blacksquare};\lambda_0)=\alpha\\
  P(\color{red}{\blacksquare};\lambda_0)=\alpha\\
  \huge{.}\\
  \huge{.}\\
  \huge{.}\\
$$

We do know that when trying to find the best test of size $\alpha$ for $H_0:\lambda=\lambda_0$ vs. $H_1:\lambda=\lambda_1$. We want all these tests at least have size $\alpha$. That mean the probability reject under each of the tests, when the null hypothesis is true, always need to be $\alpha$.

<u>When $H_1$ is true:</u>

$$
  P(\color{#33D5FF}{\blacksquare};\lambda_0)=?\\
  P(\color{orange}{\blacksquare};\lambda_0)=?\\
  P(\color{green}{\blacksquare};\lambda_0)=?\\
  P(\color{yellow}{\blacksquare};\lambda_0)=?\\
  P(\color{red}{\blacksquare};\lambda_0)=?\\
  \huge{.}\\
  \huge{.}\\
  \huge{.}\\
$$

Now we look at the alternate hypothesis, we want the numbers in question marks to be large. So the <font color='red'><b>best</b></font> test will be <font color='red'><b>largest</b></font>.

Test are defined by rejection <font color='red'><b>regions</b></font>.

For example, when $n=2$:

* Reject $H_0$ if $\overline{X}\gt2.3$
  
  $$
    \begin{align}
      &\Leftrightarrow \frac{X_1+X_2}{2}\gt2.3\\
      &\Leftrightarrow X_2\gt4.6-X_1\\
    \end{align}
  $$

Reject $H_0$ if $(X_1,X_2)$ is in this region

![png](\assets\images\notes\find-the-best-tests.png)

In general

$$
  H_0: \theta=\theta_0\quad\text{vs.}\quad H_1:\theta=\theta_1
$$

$$
  P(\color{#33D5FF}{\blacksquare};\lambda_0)=\alpha\quad P(\stackrel{\rightharpoonup}{X}\in R_1;\lambda_0)=\alpha\\
  P(\color{orange}{\blacksquare};\lambda_0)=\alpha\quad P(\stackrel{\rightharpoonup}{X}\in R_2;\lambda_0)=\alpha\\
  P(\color{green}{\blacksquare};\lambda_0)=\alpha\quad P(\stackrel{\rightharpoonup}{X}\in R_3;\lambda_0)=\alpha\\
  P(\color{yellow}{\blacksquare};\lambda_0)=\alpha\quad P(\stackrel{\rightharpoonup}{X}\in R_4;\lambda_0)=\alpha\\
  P(\color{red}{\blacksquare};\lambda_0)=\alpha\quad P(\stackrel{\rightharpoonup}{X}\in R_5;\lambda_0)=\alpha\\
  \huge{.}\\
  \huge{.}\\
  \huge{.}\\
$$

We can say that the probability that we reject in all the "colored" tests is just the probability that our vector, or our random sample $\stackrel{\rightharpoonup}{X_1}$ through $\stackrel{\rightharpoonup}{X_n}$, is in one rejection region called $R_1$ versus another rejection region called $R_2$, etc. When the null hypothesis is true and the parameter is $\lambda_0$, all of these probabilities need to be alpha.

<font color='green'><b>Definition:</b></font>

$$
  H_0:\theta=\theta_0\quad\text{vs.}H_1:\theta=\theta_1
$$

A test $R^*$ is a best test of size/level $\alpha$ for the above hypothesis if

1. $P(\stackrel{\rightharpoonup}{X}\in R^*;\theta_0)=\alpha$ and

2. If $R$ represents <u>any</u> other test with

  $$
    P(\stackrel{\rightharpoonup}{X}\in R;\theta_0)=\alpha
  $$

  then

  $$
    P(\stackrel{\rightharpoonup}{X}\in R^*;\theta_1)\ge P(\stackrel{\rightharpoonup}{X}\in R;\theta_0)
  $$

### **The Neyman-Pearson Lemma (setup)**

Let $X_1,X_2,...,X_n$ be a random sample from a distribution with pdf $f$ which depends on an unknown parameter $\theta$.

Write the joint pdf as

$$
  f(\stackrel{\rightharpoonup}{x};\theta)\stackrel{iid}{=}\prod_{i=1}^{n}f(x_i;\theta)
$$

$$
  H_0:\theta=\theta_0\quad\text{vs.}H_1:\theta=\theta_1
$$

The <font color='red'><b>best test</b></font> of size/level $\alpha$ is to reject $H_0$, in favor of $H_1$ if $\stackrel{\rightharpoonup}{x}\in R^*$ where

$$
  R^*=\Bigg\{\stackrel{\rightharpoonup}{x}:\frac{f(\stackrel{\rightharpoonup}{x};\theta_0)}{f(\stackrel{\rightharpoonup}{x};\theta_1)}\le c\Bigg\}
$$

Why does it make sense?

For discrete $X_1,X_2,...,X_n$. For the joint pdf is a joint probability

$$
  f(\stackrel{\rightharpoonup}{x};\theta)=P(X_1=x_1,X_2=x_2,...,X_n=x_n;\theta)
$$

$$
  R^*=\Bigg\{\stackrel{\rightharpoonup}{x}:\frac{f(\stackrel{\rightharpoonup}{x};\theta_0)}{f(\stackrel{\rightharpoonup}{x};\theta_1)}\le c\Bigg\}
$$

* If $H_0$ is true and $H_1$ is false, the ratio is large.

* If $H_0$ is false and $H_1$ is true, the ration is small. <font color='red'><b>This is when we should reject $H_0$!</b></font>

---

Suppose that $X_1,X_2,...,X_n$ is a random sample from the exponential distribution with rate $\lambda\gt0$.

Find the best test of size/level $\alpha$ for testing

$$
  H_0: \lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda=\lambda_1
$$

where $\lambda_1\gt\lambda_0$.

pdf: 

$$
  f(x;\lambda)=\lambda e^{-\lambda x}  
$$

joint pdf: 

$$
  \begin{align}
    f(\stackrel{\rightharpoonup}{X};\lambda)&\stackrel{iid}{=}\prod_{n=1}^{n}f(x_i;\lambda)\\
    &=\prod_{n=1}^{n}\lambda e^{-\lambda x_i}\\
    &=\lambda^ne^{-\lambda\sum_{i=1}^{n}x_i}
  \end{align}
$$

"Likelihood ratio": 

$$
  \begin{align}
    \frac{f(\stackrel{\rightharpoonup}{x};\lambda_0)}{f(\stackrel{\rightharpoonup}{x};\lambda_1)}&=\frac{\lambda_0^ne^{-\lambda_0\sum_{i=1}^{n}x_i}}{\lambda_1^ne^{-\lambda_1\sum_{i=1}^{n}x_i}}\\
    &=\bigg(\frac{\lambda_0}{\lambda_1}\bigg)e^{-(\lambda_0-\lambda_1)\sum_{i=1}^{n}x_i}
  \end{align}
$$

The Neyman-Pearson Lemma says:

* Reject $H_0$, in favor of $H_1$, if

  $$
    \bigg(\frac{\lambda_0}{\lambda_1}\bigg)^ne^{-(\lambda_0-\lambda_1)\sum_{i=1}^{n}x_i}\le c
  $$

  where $c$ is to be determined.

Let's see if we can simplify it a little bit.

The rejection rule

$$
  \bigg(\frac{\lambda_0}{\lambda_1}\bigg)e^{-(\lambda_0-\lambda_1)\sum_{i=1}^{n}x_i}\le c
$$

is equivalent to the rule

* Reject $H_0$, in favor of $H_1$, if

  $$
    e^{-(\lambda_0-\lambda_1)\sum_{i=1}^{n}x_i}\le \bigg(\frac{\lambda_0}{\lambda_1}\bigg)^nc\\
    \Downarrow\\
    e^{-(\lambda_0-\lambda_1)\sum_{i=1}^{n}x_i}\le c_1\\
  $$

Taking the log of both sides, log is a monotonically increasing function, so it's going to respect the inequality and nothing's going to flip. This is equivalent to 

$$
  -(\lambda_0-\lambda_1)\sum_{i=1}^{n}x_i\le c_2
$$

Divide both sides by $-(\lambda_0-\lambda_1)$.

Note that $\lambda_1\gt\lambda_0$, so $-(\lambda_0-\lambda_1)\gt0$.

This means that the inequality won't flip.

$$
  \sum_{i=1}^{n}x_i\le c_3
$$

In summary, the Neyman-Pearson Lemma says:

* Reject $H_0$, in favor of $H_1$, if

  $$
    \color{red}{\sum_{i=1}^{n}x_i\lt c_3}
  $$

  where $c_3$ is to be determined.

$$
  \begin{align}
    \alpha&=P(\text{Type I Error})\\
    &=P(\text{Reject }H_0;\lambda_0)\\
    &=P(\sum_{i=1}^{n}x_i\lt c_3;\lambda_0)\\
  \end{align}
$$

And we can transform to $\overline{X}$, this is equivalent to 

$$
  =P(\overline{X}\lt c_4;\lambda_0)
$$

where $c_4=c_3/n$.

And we know that:

$$
  c_4=\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
$$

<b>Conclusion</b>

The best test of size $\alpha$ for

$$
  H_0:\lambda=\lambda_0\quad\text{vs.}\quad H_1:\lambda=\lambda_1
$$

where $\lambda_1\gt\lambda_0$,

is to reject $H_0$, in favor of $H_1$ if

$$
  \overline{X}\lt\frac{\chi^2_{1-\alpha,2n}}{2n\lambda_0}
$$

---

Remember, $R^*$ is the best testof size $\alpha$ if

$$
  P(\stackrel{\rightharpoonup}{X}\in R^*;\theta_0)=\alpha\\
$$

and 

$$
  P(\stackrel{\rightharpoonup}{X}\in R^*;\theta_0)\ge P(\stackrel{\rightharpoonup}{X}\in R;\theta_1)\\
$$

for any other test of size $\alpha$.

Each of these test has its own **power function**.

$$
  \begin{align}
    \gamma_R(\theta)&=P(\text{Reject }H_0;\theta)\\
    &=P(\stackrel{\rightharpoonup}{X}\in R;\theta)\\
  \end{align}
$$

And

$$
  P(\stackrel{\rightharpoonup}{X}\in R^*;\theta_0)\ge P(\stackrel{\rightharpoonup}{X}\in R;\theta_1)\\
$$

becomes

$$
  \gamma_{R^*}(\theta_1)\ge\gamma_R(\theta_1)
$$

for any test described by R with

$$
  P(\stackrel{\rightharpoonup}{X}\in R;\theta_0)=\alpha
$$

The best test has <font color='red'><b>highest power</b></font> when $H_1$ is true.

For this reason, a best test is often referred as a most powerful test.