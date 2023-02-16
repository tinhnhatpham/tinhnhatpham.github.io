---
layout: post
comments: false
title: Axioms of Probability
categories: [Descriptive Statistics and the Axioms of Probability]
---

### **What is a Probability?**

The goal of **probability** is to assign some number, $P(A)$,
called the probability of event $A$, which will give a precise
measure to the chance that $A$ will occur. In statistics, we
draw a sample from a population, and give an estimate. So,
you will be able to understand statistics more thoroughly and
deeply if you first understand probabilities.

* Start with an experiment that generates outcomes.

* Organize all of the outcomes into a sample space, $S$.

* Let A be some event contained in $S$. That is, $A$ is some
collection of outcomes from the experiment.

What do we expect to be true of $P(A)$?

### **Axioms of Probability**

* <font color='blue'>Axiom 1</font> For any event $A, 0\le P(A)\le 1$.

* <font color='blue'>Axiom 2</font> $P(S)=1$

* <font color='blue'>Axiom 3</font> If $A_1,A_2,\ldots,A_n$ are a collection of $n$ mutually exclusive events (i.e. the intersection of any two is the empty set), then

$$
  P(\cup_{k=1}^{n}A_k)=\sum_{k=1}^{n}P(A_k)
$$

* <font color='blue'>Axiom 3 extended</font> More generally, if $A_1,A_2,\ldots$ is an infinite collection of mutually exclusive events, then

$$
  P(\cup_{k=1}^{\infty}A_k)=\sum_{k=1}^{\infty}P(A_k)
$$

These three properties are called the Axioms of Probability and we can derive many results from them.

### **Example 1**

Experiment: Flip a coin until the first tail appears. Let $0$ represent a head and $1$ a tail.

$$
  S=\{1,01,001,0001,\ldots\}
$$

Let $A_n$ represent the event of obtaining a tail on the $n^{th}$ flip, $A_n=\{00\ldots01\}$. Find $P(A_1),P(A_2),P(A_5)$ and $P(A_n)$, where $n$ is a positive integer.

$$
  \begin{align}
    &P(A_1)=1/2\\
    &P(A_2)=P(\{0,1\})=1/4\\
    &P(A_5)=P(\{00001\})=1/2^5\\
    &P(A_n)=1/2^n
  \end{align}
$$

Note: 

$$
  P(S)=P(\cup_{k=1}^{\infty}A_k)=\sum_{k=1}^{\infty}P(A_k)=\sum_{k=1}^{\infty}\frac{1}{2^k}=1
$$

If $B$ is the event that it takes at least $3$ flips to obtain a tail, find $P(B)$.

$$
  P(B)=P(\{001,0001,\ldots\})
$$

$B^c$, the complement of $B$, is the event that you obtain a tail on the first or second flip.

$$
  P(B^c)=P(\{1,01\})=1/2+1/4=3/4
$$

We also note:

$$
  \begin{align}
    &P(S)=P(B\cup B^c)=P(B)+P(B^c)=1.\text{ So,}\\
    &P(B)=1-P(B^c)=1-3/4=1/4\\
  \end{align}
$$

### **Consequenses of the Axioms**

If $A$ and $B$ are two events contained in the same sample space $S$,

* $A\cap A^c=0$ and $A\cup A^c=S$ so,

  $1 = P(S)=P(A\cup A^c)= P(A)+P(A^c)$ which implies

  $P(A^c)=1-P(A)$.

* If $A\cap B=0$ then $P(A\cap B)=0$.

* $P(A\cup B) = P(A) + P(B) - P(A\cap B)$.

These three consequenses will help us calculate many probabilities.

### **Example 2**

Select a car coming off an assembly line
and inspect it for 3 different defects (engine problem, seat
belt problem, bad paint job).

$$
  \begin{align}
    S&=\{000,100,010,001,110,101,011,111\}\\
    \mid S\mid &=8\\
  \end{align}
$$

Consider the three events:

* $A$ is the event defect 1 is present,

  $A=\{100,110,101,111\}$

* $B$ is the event defect 2 is present,

  $B=\{010,110,011,111\}$

* $C$ is the event defect 3 is present,

  $C=\{001,011,101,111\}$

Suppose over many days, data is collected and it is found that 20% of the cars have defect 1, 25% have defect 2, and 30% have defect 3. Further, 5% have defects 1 and 2, 7.5% have defects 2 and 3, 6% have defects 1 and 3, and 1.5% have all three.

$$
  \begin{align}
    &P(A)=0.2&&P(A\cap B)=0.05\\
    &P(B)=0.25&&P(B\cap C)=0.75\\
    &P(C)=0.3&&P(A\cap C)=0.06\\
    &&&P(A\cap B\cap C)=0.15\\
  \end{align}
$$

Calculate the probability of each the following events for the randomly selected car:

1. Defect 1 did not occur.

2. At least one defect occurs.

3. No defect occurs.

4. Defect 1 and 3 occur but 2 does not.

<u>Answer:</u>

1. $P(A^c) = 1 - P(A) = 1 - 0.2 = 0.8$

2. $P(\text{At least 1 defect})=P(A\cup B\cup C)$

  $=P(A)+P(B)+P(C)-P(A\cap B)-P(B\cap C)-P(A\cap C)+P(A\cap B\cap C)$

  $=0.2+0.25+0.3-0.05-0.75-0.06+0.15=0.58$

3. ![png](\assets\images\notes\2022-07-02-axioms-of-probability.png)

  $P(\text{no defect})=P(A\cup B\cup C)^c=1-P(A\cup B\cup C)$

  $=1-0.58=0.42$

4. $P(\{101\})=P(A\cap C)-P(A\cap B\cap C)=0.06-0.15=0.45$