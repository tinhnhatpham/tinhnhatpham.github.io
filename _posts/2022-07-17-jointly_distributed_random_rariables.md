---
title: Jointly Distributed Random Variables
layout: post
categories: [Probability]
---
### Jointly Discrete Distributed Random Variables
**Definition:** Given two discrete random variables, $X$ and $Y$, $p(x,y) = P(X=x)P(Y=y)$ is the **joint probability mass function** for $X$ and $Y$.

**Important property:**
* Recall: Two events $A$ and $B$, are independent if $P(A \cap B) = P(A)P(B)$.
* $X$ and $Y$ are **independent random variables** if $P(X=x, Y=y) = P(X=x)P(Y=y)$ for all possible values of $x$ and $y$.

### Jointly Continous Distributed Random Variables
**Definition:** If $X$ and $Y$ are continous random variables, then $f(x,y)$ is the **joint density mass function** for $X$ and $Y$ if:
* $P(a \le X \le b, c \le Y \le d) = \int_{a}^{b}\int_{c}^{d}f(x,y)dxdy$ for all possible $a$, $b$, $c$, and $d$.

**Important property:**
* $X$ and $Y$ are **independent random variable** if $f(x,y) = f(x)f(y)$ for all possible values of $x$ and $y$.

