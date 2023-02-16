---
layout: post
comments: false
title: Counting - Permutations and Combinations
categories: [Descriptive Statistics and the Axioms of Probability]
---

### **Counting**

The goal of **probability** is to assign some number, $P(A)$, called the probability of event $A$, which will give a precise measure to the chance that $A$ will occur. If a sample space, $S$, has $N$ single events, and if each of these events is equally likely to occur, then we need only count the number of events to find the probability.

For example, if $S=\{E_1,E_2,\ldots,E_N\}$ and if $P(E_k) = 1/N$ for $k=1,2,\ldots,N$, and if $A$ is an event in $S$, then

$$
  P(A)=\frac{\text{number of simple events in }A}{N}
$$

**Example**

Experiment: Roll a six-sided dice twice.

$S=\{(i,j)\mid i,j \in \{1,2,3,4,5,6\}\}$, $\mid S\mid=36$ and each of the $36$ outcomes of $S$ is equally likely.

* Let $A$ be the event of rolling a $1$ on the first roll.

  $P(A)=P(\{11,12,13,14,15,16\})=\frac{6}{36}=\frac{1}{6}$

* Let $B$ be the event that the sum of the two rolls is $8$.

  $P(B)=P(\{26,35,44,53,62\})=\frac{5}{36}$

* Let $C$ be the event that the value of the second roll is two more that the first roll.

  $P(C)=P(\{13,24,35,46\})=\frac{4}{36}=\frac{1}{9}$

---

### **Permutations**

Any **ordered** sequence of $k$ objects taken from a set of $n$ distict objects is called a **permutation of size** $k$.

Notation: $P_{k,n}$

**Example:** Suppose an organization has $60$ members. One person is selected at random to be the president, another person is seleted as vice-president, and a third is selected as the treasurer. How many ways can it be done? (This would be the cardinality of the sample space.)

$$
  P_{3,60}=60.59.58=\frac{60!}{57!}=205,320
$$

<u>Definition:</u> $n!=n(n-1)(n-2)...3.2.1$ for any positive integer $n$. By definition, we take $0!=1$.

---

### **Combinations**

Given $n$ distinct objects, any unordered subset of size $k$ of the objects is call **combination**. 

Notation: $C_{k,n}$

**Example 1** Suppose we have $60$ people and want to choose a $3$ person team (order is not important). How many combinations are possible?

Sample space of permutations

$$
  S_p = \left.
  \begin{cases}
    123&,124&,125&,\ldots\\
    132&,142&,\ldots&,\ldots\\
    213&,214&,\ldots&,\ldots\\
    231&,241&,\ldots&,\ldots\\
    312&,412&,\ldots&,\ldots\\
    321&,421&,\ldots&,\ldots\\
  \end{cases}
  \right\}\quad\mid S\mid=\frac{60!}{57!}
$$

Combinations: 

$$
  S_c=\{123,124,125,\ldots\}\quad\mid S\mid=\frac{60!}{57!3!}={60\choose3}
$$

Note: ${60\choose3}={60\choose57}$

Notation: 

$$
  \color{red}{
    {n\choose k}=\frac{n!}{k!(n-k)!}
  }
$$

This represents the number of combinations of size $k$ chosen from $n$ distinct objects.

**Example 2** Suppose we have the same $60$ people, $35$ are female and $25$ are male. We need to select a committee of $11$ people.

* How many ways can such a committee be formed?

  $$
    \text{# of committees 11} = {60\choose11}=\frac{60!}{11!(60-11)!}\\
    \mid S\mid={60\choose11}
  $$

* What is the probability that a randomly selected committe will contain at least $5$ men and at least $5$ women? (Assume each committee is equally likely).

  $$
    \begin{align}
      &P(\text{at least 5M and at least 5W on committee})\\
      &=P(5M+6W) + P(6M+5W)\\
      &=\frac{\binom{25}{5}\binom{35}{6}}{\binom{60}{11}}+\frac{\binom{25}{6}\binom{35}{5}}{\binom{60}{11}}
    \end{align}
  $$

**Example 3**

A city has brought $20$ buses. Shortly after being put into service, some of them develop cracks in the frame. The buses are inspected and $8$ have visible cracks.

* How many way can the city select a sample of $5$ for thorough inspection? (Assume each bus is equally likely to be chosen.)

  $$
    \mid S\mid={20\choose5}
  $$

* If $5$ buses are chosen at random, find the probability that exactly $4$ have cracks.

  $$
    P(\text{4 with cracks})=\frac{\binom{12}{1}\binom{8}{4}}{20\choose5}
  $$

* If $5$ buses are chosen at random, find the probability that at least $4$ have cracks.

  $$
    P(\text{at least 4 with cracks})\\
    =\frac{\binom{12}{1}\binom{8}{4}}{20\choose5}+\frac{\binom{12}{0}\binom{8}{5}}{20\choose5}
  $$