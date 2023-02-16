---
layout: post
comments: false
title: Binomial and Negative Binomial Random Variables
categories: [Discrete Random Variable]
---

### **Binomial Random Variable**

Examples:

* Suppose you toss a fair coin 12 times. What is the
probability that you'll get 5 heads?

* Suppose you pick a random sample of 25 circuit boards
used in the manufacture of a particular cell phone. You
know that the long run percentage of defective boards is
5%. What is the probability that 3 or more boards are
defective?

* Suppose 40% of online purchasers of a particular book
would like a new copy and 60% want a used copy. What
is the probability that amongst 100 random purchasers,
50 or more used books are sold?

These three situations, and many more, can be modeled by a binomial random variable.

Key elements of these examples above:

* $n$ Bernoulli trials.
* Same probability of success on each trial $(p=0.5, p=0.05, p=0.6)$.
* These are independent Bernoulli trials (one trial does not affect the outcome of another).

---

Properties of a binomial random variable:

* Experiment is $n$ trials ($n$ is fixed in advance).
* Trials are identical and resultin a success or a failure (i.e. Bernoulli trials) with $P(\text{success})=p$ and $P(\text{failure})=1-p$.
*Trials are independent (outcome of one trial does not influence any other).

If $X$ is the number of success in the $n$ independent and identical trials, $X$ is a binomial random variable.

Notation: $\color{red}{X\sim\text{Bin}(n,p)}$

--- 

Find the pmf, expectation, and variance for a binomial random variable, $X\sim\text{Bin}(n,p)$

What is the sample space for a binomial experiment?

$$
  S=\{(x_1,x_2,\ldots,x_n\}\\
  x_i=\begin{cases}
    \begin{align}
      &1&&\text{if success on i^{th} trial}\\
      &0&&\text{if failure}\\
    \end{align}
  \end{cases}\\
  \vert S\vert=2^n
$$

<font color='blue'><b>PMF:</b></font>

$$
  \begin{align}
    P(X=0)&=P(\{0...\})=(1-p)^n\\
    P(X=1)&=P(\{10..0,010..0,...,00..01\})=np(1-p)^{n-1}\\
    P(X=2)&=P(\{\underbrace{110..0}_{\text{2 1's + (n-2) 0's}},...\})=\binom{n}{2}p^2(1-p)^{n-2}\\
    P(X=k)&=P({\underbrace{...}_{\text{k 1's}\\\text{n-k 0's}},...})=\color{red}{\binom{n}{k}p^k(1-p)^{n-k}},\quad k=0,1,...\\
  \end{align}
$$

<font color='blue'><b>Expected value:</b></font>

**Definition:** The expected value of a discrete random variable, $E(X)$, is given by

$$
  E(X)=\sum_{k}kP(X=k)
$$

$X\sim\text{Bin}(n,p)$

$$
  E(X)=\sum_{k=0}^{n}k\binom{n}{k}p^k(1-p)^{n-k}=\color{red}{np}
$$

<u>Recall:</u> 

$\text{Bern}(p)$ has expected value $p$, $X_1,X_2,\ldots X_n$ as independent$\text{Bern}(p)$

$$
  X = \sum_{k=1}^{n}X_n\\
  E(X) = \sum_{k=1}^{n}E(X_n) = np
$$

<font color='blue'><b>Variance:</b></font>

**Definition:** The **variance** of a random variable is given by 

$\sigma_X^2=V(X)=E[(X-E(X))^2]$.

Computational formula: $V(X)=E(X^2)-(E(X))^2$.

$X\sim\text{Bin}(n,p)$

$$
  \begin{align}
    V(X)&=\sum_{k=0}^{n}\bigg(k-E(X)\bigg)^2\binom{n}{k}p^k(1-p)^{n-k}\\
    &=\color{red}{n\times\underbrace{p(1-p)}_{\text{variance of Bern($p$)}}}
  \end{align}
$$

### **Negative Binomial Random Variable**

Examples:

* Suppose you toss a fair coin until you obtain 5 heads.
How many tails before the fifth head?

* Suppose you randomly choose circuit boards until you
find 3 defectives. You know that the long run percentage
of defective boards is 5%. How many must you examine?

* Suppose 40% of online purchasers of a particular book
would like a new copy and 60% want a used copy. How
many new books are sold before the fiftieth used book?

These three situations can be modeled by a <font color='red'><b>negative</b></font> binomial
random variable.

Key elements for these examples above:

* Independent Bernoulli trials until $r$ successes.
* Count the number of failure until $r^{th}$ success.

---

**Definition:** Repeate independent Bernoulli trials until a total of $r$ successes is obtained. The negative binomial random variable $Y$ counts the number of failures before the $r^{th}$ success. Notation: $Y\sim\text{NB}(r,p)$.

* The number of successes $r$ is fixed in advance.
* Trials are identical and result in a success or a failure (i.e. Bernoulli trials) with $P(\text{success})=p$ and $P(\text{failure})=1-p$.
* Trials are independent (outcome of one trial does not influence any other).

Compare to $X\sim\text{Bin}(n,p)$: $X$ is the number of successes in the $n$ independent and identical trials and $n$ is fixed in advance.

---

Example: A physician wishes to recruit 5 people to participate in a medical study. Let $p=0.2$ be the probability that a randomly selected person agrees to participate. What is the probability that 15 people must asked before 5 are found who agree to participate.

$Y$ is the number of failures before the 5 people are found.

$$
  S=\{(x_1,x_2,\ldots)\}\\
  \begin{cases}
    \begin{align}
      &1&&\text{if $S$ on the $i^{th}$ trial}\\
      &0&&\text{if failure}\\
    \end{align}
  \end{cases}\\
    \text{and }\sum x_i=5
$$

We have

$$
  \begin{align}
    P(Y=0)&=P(\{\})=(0.2)^5\\
    P(Y=1)&=P(\{011111,101111,...,111101\})=\binom{5}{4}(0.2)^5(0.8)^1\\
    P(Y=2)&=\binom{6}{4}(0.2)^5(0.8)^2
  \end{align}
$$

In general

$$
  P(Y=K)=\binom{k+5-1}{4}(0.2)^5(0.8)^k
$$

---

**Summary**

If $Y\sim\text{NP}(r,p)$, we have

$$
  \color{red}{P(Y=k)=\binom{k+r-1}{r-1}p^r(1-p)^k\quad\text{for }k=0,1,2...}
$$

$$
  \begin{align}
    E(Y)&=\frac{r(1-p)}{p}\\
    V(Y)&=\frac{r(1-p)}{p^2}
  \end{align}
$$

Relationship between geometric and negative binomial random variables?

$X\sim\text{Geom}(p)\leftarrow$ repeat independent identical Bernoulli trials until $\color{red}{1^{st}}$ success.

$Y\sim\text{NP}(1,p)\leftarrow$ count the number of failures until $\color{red}{1^{st}}$ success.

Because $Y$ counts the number of failures until the first success, so $Y$ is the same as $X - 1$.

$Y=X-1$ then $E(Y)=E(X)-1=\frac{1}{p}-1=\frac{1-p}{p}$

For the $\text{NB}(r,p)$, we have $r$ $\text{Geom}(p)$ stack one to another.

$$
  \text{NB}(r,p)=\underbrace{....}_{F}1\underbrace{.....}_{F}1\underbrace{.....}_{F}\overbrace{\color{red}{1}}^{r^{th}\text{ success}}
$$

Then we have 

$$
  \begin{align}
    E(Y)&=\frac{r(1-p)}{p}\\
    V(Y)&=\frac{r(1-p)}{p^2}
  \end{align}
$$