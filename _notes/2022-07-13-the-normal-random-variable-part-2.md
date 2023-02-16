---
layout: post
comments: false
title: The Gaussian (normal) Random Variable Part 2
categories: [Continuous Random Variables]
---

If $X\sim N(\mu,\sigma^2)$ then the pdf:

$$
  \color{red}{f_X(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}}\quad\text{for }-\infty\lt x\lt\infty
$$

If $Z\sim N(0,1)$ then the pdf:

$$
  \color{red}{f_Z(x)=\frac{1}{\sqrt{2\pi}e^{-x^2/2}}}\quad\text{for }-\infty\lt x\lt\infty
$$

**Proposition:** If $X\sim N(\mu,\sigma^2)$, then $\frac{X-\mu}{\sigma}\sim N(0,1)$

We can think of $\frac{X-\mu}{\sigma}$ as $X$ shifted by $\mu$, scaled by $\frac{1}{\sigma}$.

<u>Aside:</u> A continuous rv $Y$ with density function $f_Y(y)$.

* We know $P(Y\le a)=\int_{-\infty}^{a}f_Y(y)dy$.

* And $P(2Y\le a)=P(Y\le\frac{a}{2})=\int_{-\infty}^{a/2}f_Y(y)dy$

<u>Proof:</u>

$$
  \begin{align}
    P(\frac{x-\mu}{\sigma}\le a)&=P(X\le a\sigma+\mu)\\
    &=\int_{-\infty}^{a\sigma+\mu}\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}\\
  \end{align}
$$

Let $u=\frac{x-\mu}{\sigma}$ and $du=\frac{1}{6}dx$.

$$
  \int_{-\infty}^{a}\underbrace{\frac{1}{\sqrt{2\pi}}e^{-u^2/2}}_{\text{density function for }N(0,1)}du
$$

---

**Example 1:** If $X\sim N(1,4)$

a. Find $P(0\le X\le3.2)$.

b. Find $a$ so that $P(X\le a)=0.7$.

Solutions:

a. 

$$
  \begin{align}
    P(0\le X\le3.2)&=\int_0^{3.2}f_X(x)dx\\
    &=P\bigg(\frac{0-1}{2}\le\underbrace{\frac{X-1}{2}}_{Z}\le\frac{3.2-1}{2}\bigg)\\
    &=P\bigg(-\frac{1}{2}\le Z\le1.1\bigg)\\
    &=P\bigg(Z\le1.1)-P(Z\lt-\frac{1}{2}\bigg)\\
    &=\Phi(1.1)-\Phi(-\frac{1}{2})\\
    &\approx0.5558
  \end{align}
$$

b.

![png](\assets\images\notes\the-normal-random-variable-part-2-1.png)

We can use the table above to find $P(X\le a)=0.7$.

$$
  P\bigg(\frac{X-1}{2}\le\frac{a-1}{2}\bigg)=P\bigg(Z\le\frac{a-1}{2}\bigg)=0.7
$$

From the table we have:

$$
  \Phi(0.52)=0.6985\\
  \Phi(0.53)=0.7019\\
$$

$0.7$ is about mid-way between these two values above.

$$
  \begin{align}
    &\Rightarrow\Phi(0.525)\approx0.7\\
    &\Rightarrow\frac{a-1}{2}=0.525\\
    &\Rightarrow a=2.05
  \end{align}
$$

Therefore,

$$
  P(X\le2.05)=0.7
$$

**Example 2:** The time that it takes a driver to react to the brake lights on a decelerating vehicle is critical in helping to avoid rear-end collisions. Research suggests that reaction time for an in-traffic response to a brake signal from standard brake lights can be modeled with a normal distribution having mean $1.25$ seconds and standard deviation 0.46 seconds. What is the
probability that the reaction time is between $1$ and $1.75$
seconds? 

Let $X$ = reaction time, $X\sim N(1.25, 0.46^2)$.

$$
  \begin{align}
    P(1\le X\le1.75)&=P\bigg(\frac{1-1.25}{0.46}\le\frac{X-1.25}{0.46}\le\frac{1.75-1.25}{0.46}\bigg)\\
    &=P(-0.543\le Z\le1.087)\\
    &=\Phi(1.087)-\Phi(-0.543)\\
    &\approx0.568
  \end{align}
$$

---

Normal approximation to the binomial distribution

Recall: $X\sim\text{Bin}(n,p)$ means that $X$ couts the number of successes in $n$ Bernoulli trials, each with probability of success $p$.

$$
  P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}\quad\text{for }k=0,1,...,n.\\
  E(X)=np\quad\text{and}\quad V(X)=np(1-p)\\
$$

For large $n$, $X$ can be approximated by a normal rv with

$$
  \mu=np\quad\text{and}\quad\sigma^2=np(1-p)
$$

If $X\sim\text{Bin}(n,p)$ and $np(1-p)\ge10$ then

$$
  \frac{X-np}{\sqrt{np(1-p)}}\approx N(0,1)
$$

---

**Example 3:** In a given day, there are approximately $1,000$ visitors to a website. Of these, $25\%$ register for a service. Estimate the probability that between $200$ and $225$ people will register for a service tomorrow.

Let $X$ = # of people who register for a service

$X\sim\text{Bin}(n=1000,p=0.25)$

We can calculate the probability by:

$$
  P(200\le X\le225)=\sum_{k=200}^{225}\binom{1000}{k}p^k(1-p)^{1000-k}
$$

But it's incredible to calculate and compute directly. So we're going to convert it to normal random variable:

Use $\mu=np=250$, $\sigma^2=np(1-p)=187.5$. We have

$$
  P\bigg(\frac{199.5-250}{\sqrt{187.5}}\le\frac{X-250}{\sqrt{187.5}}\le\frac{225.5-250}{\sqrt{187.5}}\bigg)
$$

The reason we add $0.5$ on the right and subtract $0.5$ on the left to accomondate for the fact that the binomial is discrete, and we're approximating it by a continuous distribution. If not we might loose some values when calculating. It's called **continuity correction**. 

$$
  \begin{align}
    &P\bigg(\frac{199.5-250}{\sqrt{187.5}}\le\frac{X-250}{\sqrt{187.5}}\le\frac{225.5-250}{\sqrt{187.5}}\bigg)\\
    &=P(-3.688\le Z\le-1.789)\\
    &=\Phi(-1.789)-\Phi(-3.688)\\
    &\approx0.0367
  \end{align}
$$

In R:

```R
pnorm(-1.789)-pnorm(-3.688)
```
