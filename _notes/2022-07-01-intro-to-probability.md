---
layout: post
comments: false
title: Intro to Probability
categories: [Descriptive Statistics and the Axioms of Probability]
---

### **What is statistic?**

**Statistics is the science of using data effectively to gain new knowledge.** We need data to learn something new. We
need to collect and analyze the data ethically.

**Population**: Those individuals or objects from which we want
to acquire information or draw a conclusion. Most of the time,
the population is so large, we can only collect data on a subset
of it. We will call this our **sample**.

In **probability** we assume we know the characteristics of the
entire population. Then, we can pose and answer questions
about the nature of a sample. In **statistics**, if we have a
sample with particular characteristics, we want to be able to
say, with some degree of confidence, whether the whole
population has this characteristic, or not.

**Probability** studies randomness and uncertainty by giving
these concepts a mathematical foundation.
For example, we want to understand how to find the
probability

* of getting at least 2 heads in 5 coin flips,

* that a customer will buy milk if they are also buying
bread,

* that the price of a stock will be in a certain range on a
certain date in the future.

Probability gives us the framework to quantify uncertainty.

### **Terminology**

* An **experiment** is any action or process that generates
observations.

* The **sample space** of an experiment, denoted S, is the
set of all possible outcomes of an experiment.

* An **event** is any possible outcome, or combination of
outcomes, of an experiment.

* The **cardinality** of a sample space or an event, is the
number of outcomes it contains. $\mid S\mid$ represents the
cardinality of the sample space.

### **Examples**

For each of the following, describe the sample space, S, and
give its cardinality.

* Experiment 1: Flip a coin once

$$
  \begin{align}
    S &= \{H,T\} = \{0,1\}\quad\begin{cases}
      0=\text{head}\\
      1=\text{tail}\\
    \end{cases}\\
    \mid S\mid &=2\\
  \end{align}
$$

* Experiment 2: Flip a coin twice

$$
  \begin{align}
    S &= \{00,01,10,11\}\quad\begin{cases}
      0=\text{head}\\
      1=\text{tail}\\
    \end{cases}\\
    \mid S\mid &=4\\
  \end{align}
$$

* Experiment 3: Flip a coin until you get a tail.

$$
  \begin{align}
    S&=\{1,01,001,0001,...\}\\
    \mid S\mid &=\infty\\
  \end{align}
$$

* Experiment 4: Select a car coming off an assembly line
and inspect it for 3 different defects (engine problem, seat
belt problem, bad paint job).

$$
  \begin{align}
    S&=\{000,100,010,001,110,101,011,111\}\\
    \mid S\mid &=8\\
  \end{align}
$$

### **Set Notation**

For events $A$ and $B$,

* $A\cup B$, the **union** of $A$ and $B$, means an outcome in $A$ or $B$ occurs.

* $A\cap B$, the **intersection** of $A$ and $B$, is all the outcomes that are in both $A$ and $B$.

* $A^c$, the **compliment** of $A$, means the set of all events in $S$ that are not in $A$.

* $A$ and $B$ are mutually exclusive, or disjoint, if they have no events in common. We write $A\cap B=0$.

### **Examples continued**

$S=\{000,100,010,001,110,101,011,111\}$ Consider the following events:

* $A$ is the event that there is an engine problem (defect 1). In set notation: $A=\{100,110,101,111\}$.

* $B$ is the event that there is exactly one defect. In set notation: $B=\{100,010,001\}$.

* $C$ is the event that there are exactly two defects, so $C=\{110,101,011\}$.

* $A\cap B=\{100\}$

* $A^c=\{000,010,001,011\}$

* $A^c\cup B=\{000,010,001,011,100\}$

* $B\cap C=0$

### **Venn Diagrams**

Venn diagrams can be used to help us visualize unions, intersections, and complements.

![png](\assets\images\notes\2022-07-01-intro-to-probability.png)