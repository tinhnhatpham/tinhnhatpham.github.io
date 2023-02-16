---
layout: post
comments: false
title: Independent Events
categories: [Conditional Probability]
---

### **Independence**

Two events are **independent** if knowing the outcome of one
event does not change the probability of the other.

Examples:

* Flip a two-sided coin repeatedly. Knowing the outcome of
one flip does not change the probability of the next.
* Roll a dice repeatedly.
* What about polling? What if you ask two randomly
selected people about their political affiliation? What if
the two people are friends? In this case, it's not independent!

### **Definition**

Two events, $A$ and $B$, are **independent** if $P(A\vert B)=P(A)$, or equivalently, if $P(B\vert A) = P(B)$.

We have that

$$
  P(A\vert B)=\frac{P(A\cap B)}{P(B)}
$$

then, if $A$ and $B$ are independent, we get the multiplication rule for independent events:

$$
  P(A\cap B) = P(A)P(B)
$$

**Definition** Events $A_1,\ldots,A_n$ are **mutually exclusive independent** if for every $k$ $(k=2,3,...n)$ and every subset of indices $i_1,i_2,\ldots,i_k$:

$$
  P(A_{i_1}\cap A_{i_2}\cap\ldots\cap A_{i_n})=P(A_{i_1})P(A_{i_2})\ldots P(A_{i_k})
$$

Use the definition of independence in two ways:

* We can use the definition to show two events $A$ and $B$ are (or are not) independent. To do this, we calculate $P(A)$, $P(B)$, and $P(A\cap B)$ to check if $P(A\cap B)=P(A)P(B)$.
* If we know two events are independent, we can find the probability of their intersection.

### **Examples**

---

**Example 1**

Roll a six-sided dice twice. We have

$$
  S=\{(i,j)\vert i,j\in{1,2,3,4,5,6}\}\\
  \vert S\vert=36
$$

and each of the $36$ outcomes of $S$ is equally likely.

Let $E$ be the event that the sum is $7$.

Let $F$ be the event that the first roll is a $4$.

Let $G$ be the event that the second roll is a $3$.

What can you say about the independence of $E$, $F$, and $G$?

$$
  P(E)=P(\{16,25,34,43,52,61\})=1/6\\
  P(F)=P(\{41,42,43,44,45,46\})=1/6\\
  P(G)=P(\{13,23,33,43,53,63\})=1/6\\
$$

$$
  P(E\cap F)=P(\{43\})=1/36=P(E)P(F)\\
  P(E\cap G)=P(\{43\})=1/36=P(E)P(F)\\
  P(F\cap G)=P(\{43\})=1/36=P(F)P(G)\\
$$

We say that any pair of $E$, $F$, or $G$ is pairwise independent. Now we check for mutually independent of these three:

$$
  \underbrace{P(E\cap F\cap G)}_{1/36}\color{red}{\ne}\underbrace{P(E)P(F)P(G)}_{(1/6)^3}
$$

---

**Example 2**

In a school of 1200 students, 250 are juniors, 150 students are
taking a statistics course, and 40 students are juniors and also
taking statistics. One student is selected at random from the
entire school. Let $J$ be the event the selected student is a
junior. Let $S$ be the event that the selected student is taking
statistics.

If the randomly chosen student is a junior, then what is the
probability that they are also taking stats? Are $J$ and $S$
independent?

Let 

$$
  \begin{align}
    &J=\text{Junior}&&S=\text{stats}\\
    &P(J)=\frac{250}{1200}&&P(S)=\frac{150}{1200}\\
    &P(S\cap J)=\frac{40}{1200}
  \end{align}
$$

1. What is the probability that they are taking stats give the student is a junior:

  $$
    P(S\vert J)=\frac{P(S\cap J)}{P(J)}=\frac{40/1200}{250/1200}=0.16
  $$

2. Are $J$ and $S$ independent?

  $$
    P(S\vert J)\ne P(S)\rightarrow\text{Not independent!}
  $$

  Also note:

  $$
    P(S).P(J)=\frac{150}{1200}\frac{250}{1200}\ne P(S\cap J)=\frac{40}{1200}
  $$

---

**Example 3**

Suppose you have a system of components as in the diagram. Let $A_i$ be the event that the $i^{th}$ component works and assume $P(A_i)=0.9$ for $i=1,2,3,4,5$. Assume the components work independently of each other. For the system to work, you need a path of working components from the start to finish. Find the probability that the system works.

![png](\assets\images\notes\2022-07-05-independent-events.png)

$$
  \begin{align}
  &S=\{(x_1,x_2,x_3,x_4,x_5)\}\\
  &\begin{cases}
    x_i=1\quad\text{if $i^{th}$ component works}\\
    x_i=0\quad\text{if $i^{th}$ component doesn't works}\\
  \end{cases}
  \end{align}
$$

$\vert S\vert=2^5=32$ but each element in $S$ is not equally likely. For example:

$$
  P(00000)=(0.1)^5 \ne P(10101)=(0.9)^3(0.1)^2
$$

Recall:

$$
  P(A\cup B\cup C) = \\
  P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C)\quad\text{(1)}
$$

The probability that the system works:

$$
  \begin{align}
    P&(\text{system works})\\
    &=P(A_1\cap A_2)\cup(A_1\cap A_3)\cup(A_4\cap A_5)\\
    &\qquad\text{apply (1), we have:}\\
    &=P(A_1\cap A_2)+P(A_1\cap A_3)+P(A_4\cap A_5)\\
    &\qquad-P(A_1\cap A_2\cap A_3)-P(A_1\cap A_2\cap A_4\cap A_5)-P(A_1\cap A_3\cap A_4\cap A_5)\\
    &\qquad+P(A_1\cap A_2\cap A_3\cap A_4\cap A_5)\\
    &=3(0.9)^2-(0.9)^3-2(0.9)^4+(0.9)^5\\
    &=0.9792\quad\text{(overall prob that system works)}
  \end{align}
$$

If there only two components work $(P(A_1\cap A_2)=(0.9)^2=0.81)$, the overall probability decreased. So the key to increasing probability system works is redundancy!

<u>One final question:</u> Suppose you know two events $A$ and $B$ are mutually exclusive, that is, $A\cap B=0$. Are $A$ and $B$ independent?

$$
  P(A\vert B)=\frac{P(A\cap B)}{P(B)}=0\quad\text{(since $A\cap B = 0)$}\\
$$

Knowing $B$ has occurred means $A$ cannot occur. So $A$ and $B$ are dependent!