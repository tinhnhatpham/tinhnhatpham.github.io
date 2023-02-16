---
layout: post
comments: false
title: Conditional Probability and Bayes Theorem
categories: [Conditional Probability]
---

### **Conditional Probability**

Suppose we have two event $A$ and $B$ from the sample space $S$. We want to calculate the probability of event $A$, knowing that event $B$ has occured. $B$ is the "conditional event". Notation: $P(A\vert B)$, the probability of event $A$ given that $B$ has occured.

<u>Example</u>: Roll a six-sided dice twice. We have

$S =\{(i,j)\mid i,j\in{1,2,3,4,5,6}\},\vert S\vert=36$ and each of the $36$ outcomes of $S$ is equally likely.

Let $A$ be the event that at least one of the dice shows a $3$.

* $A=\{(3,1),(3,2),\ldots,(3,6),(1,3),(2,3),\ldots,(6,3)\}$

* $P(A)=11/36$

Let $B$ be the event that the sum of the $2$ dice is $9$.

* $B=\{(6,3),(3,6),(4,5),(5,4)\}$

* $P(B)=4/36$

<i>Question:</i> Suppose we know that $B$ has occured. How does this change the probability of $A$? That is, find $P(A\vert B)$, the probability that at least one dice was a $3$ given that the sum was $9$.

* $P(A\cup B)=P((3,6),(6,3))=2/36$

* $P(A\vert B)=\frac{P(A\cap B)}{P(B)}=\frac{2/36}{4/36}=1/2$

![png](\assets\images\notes\2022-07-04-conditional-probability-and-bayes-theorem.png)

---

### **Bayes Theorem**

**Conditional probability** is defined as:

$$
  P(A\vert B)=\frac{P(A\cap B)}{P(B)},\quad P(B)\gt0
$$

This leads to the **multiplication rule**

$$
  P(A\cap B) = P(B)P(A\vert B)
$$

Similarly,

$$
  P(B\cap A) = P(A)P(B\vert A)
$$

**Bayes Theorem:** Let $P(B)\gt0$. Then,

$$
  P(A\vert B)=\frac{P(A)P(B\vert A)}{P(B)}
$$

---

### **Law of Total Probability**

Give two events $A$ and $B$ from the same sample space,

$$
  B=(B\cap A)\cup(B\cap A^c)\\
$$

We have 

$$
  \begin{align}
    P(B)&=P(B\cap A) + P(B\cap A^c)\\
    &=P(B\vert A)P(A)+P(B\vert A^c)P(A^c)
  \end{align}
$$

![png](\assets\images\notes\2022-07-04-conditional-probability-and-bayes-theorem-1.png)

Extend this idea to $n$ sets $A_1,A_2,\ldots,A_n$ where 

$A_1\cap\dots A_n=0$ and $\cup_{k=1}^{n}A_k=S$. 

Note: $A_1,A_2,\ldots,A_n$ are mutually exclusive means $A_i\cap A_j=0$ for all $i,j(i\ne j)$.

![png](\assets\images\notes\2022-07-04-conditional-probability-and-bayes-theorem-2.png)

We have

$$
  \begin{align}
  P(B) &= P(B\cap A_1)+P(B\cap A_2)+\\
  &\quad P(B\cap A_3) + \underbrace{P(B\cap A_4)}_{0}\\
  &=P(B\vert A_1)P(A_1)+P(B\vert A_2)P(A_2)+P(B\vert A_3)P(A_3)
  \end{align}
$$

Then,

$$
  P(B)=\sum_{k=1}^{n}P(B\vert A_k)P(A_k)
$$

---

### **Example - Testing for a disease**

<u>Example:</u> Suppose your compant has developed a new test for a disease. Let event $A$ be the event that a randomly selected individual has the disease and, from other data, you know that 1 in 1000 people has disease. Thus, $P(A)=0.001$. Let $B$ be the event that a positive test result is received for the randomly selected individual. Your company collects data on their new test and finds the following:

* $P(B\vert A)=0.99\rightarrow P(\text{pos test result}\vert\text{person has the disease})$

* $P(B^c\vert A)=0.01\rightarrow P(\text{neg test result}\vert\text{person has the disease})$

* $P(B\vert A^c)=0.02\rightarrow P(\text{pos test result}\vert\text{person doesn't have the disease})$

Calculate the probability that the person has the disease, given a positive test result. That is, find $P(A\vert B)$.

$$
  \begin{align}
    P(A\vert B)&=\frac{P(A\cap B)}{P(B)}\\
    &=\frac{P(B\vert A)P(A)}{P(B)}\quad\text{(Bayes Theorem)}\\
    &=\frac{P(B\vert A)P(A)}{P(B\vert A)P(A)+P(B\vert A^c)P(A^c)}\quad\text{(Law of total prob.)}\\
    &=\frac{(0.99)(0.001)}{(0.99)(0.001)+(0.02)(0.999)}\\
    &=0.0472
  \end{align}
$$

* $P(A)=0.001\leftarrow$ prior probability of $A$.
* $P(A\vert B)=0.0472\leftarrow$ posterior probability of $A$.

### **Example - Tree Diagram**

![png](\assets\images\notes\2022-07-04-conditional-probability-and-bayes-theorem-3.png)

Sample space: 

$$
  \begin{align}
    D&=\text{disease}, &&+=\text{pos. test result}\\
    N&=\text{no disease}, &&-=\text{neg. test result}
  \end{align}
$$

$$
  S=\{(D,+),(D,-),(N,+),(N,-)\}
$$