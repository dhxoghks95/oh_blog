---
title: A First Course in Bayesian Statistical Methods Chapter 3 . One parameter model
subtitle : 1. Binomial Model
date: 2021-03-08T20:50:13+09:00
categories: ["A First Course In Bayesian Statistical Methods"]
tags: ["Bayesian", "Python"]
---

# A First Course in Bayesian Statistical Methods

## 3. 파라미터가 하나인 모델(One-parameter models)

**One parameter model**

하나의 미지의 파라미터에 의해 결정되는 표본 분포(sampling distribution)

이번 챕터에서 다룰 분포의 종류

- 이항 모델(Binomial model)
- 포아송 모델(Poisson model)

이번 챕터에서는 다음과 같은 세 가지 기본적인 베이지안 데이터 분석을 배워볼 것입니다.

- 켤레 사전 분포(Conjugate Prior Distribution)
- 사후 예측 분포(Posterior Predictive Distribution)
- 신뢰 영역(Confidence Region : 신뢰구간의 다차원 버전)

## 3.1 이항 모델(The binomial model)

### 행복 데이터

대상 : 65세 이상 여성 129명(exchangeable)

$$
Y_i = \begin{cases} 1 \ \ \ \text{i 번째 응답자가 행복이라고 답함} \newline 0 \ \ \ \text{그 외} \end{cases}
$$

이 때, 샘플 129개가 모집단의 크기 $N$보다 작으므로, 이전 장에서 배운 것 처럼 $Y_1, ..., Y_{129}$는 다음과 같이 잘 근사됩니다.

- $\theta$에 대한 믿음 = $\sum^n_{i=1} Y_i / N$;
- $Y_i|\theta$ = 기댓값이 $\theta$이고 i.i.d인 이항 확률 변수(binomial random variable)

즉 {$y_i, ..., y_{129}$}의 잠재적 산출물의 확률은 다음과 같습니다.

$$
p(y_1, ...., y_{129}) | \theta) = p(y_1|\theta) \ p(y_2|\theta) \ ... \ p(y_{129}|\theta)
$$



$$
= \prod^{129}_{i=1} \theta^{y_i} (1 - \theta)^{1 - y_i} = \theta^{\sum^{129}_{i=1}y_i}(1-\theta)^{129 - \sum^{129}_{i=1}y_i}
$$

이제 필요한 것은 사전 분포입니다.

### 균등 사전 분포(Uniform prior distribution)

$\theta$ : 0과 1 사이의 값

여기서, 우리의 사전 정보를 [0,1]에서 같은 길이와 같은 확률을 가지는 부분 구간(subinterval)로 가정합시다. 수식으로 표현하면 다음과 같습니다.

$$
Pr(a \leq \theta \leq b) = Pr( a+c \leq \theta \leq b+c) \ \ for \ \ 0 \leq a < b < b+c \leq 1
$$

위 식은 c에 의해 구간의 위치가 변하더라도, 0과 1 사이에서 길이가 b-a라면 같은 확률을 가진다는 것을 의미합니다.

이러한 균등 확률 밀도 함수는 다음과 같이 쓸 수 있습니다 :

$$
p(\theta) = 1 \text{ for all} \ \theta \in [0,1].
$$

베이즈의 법칙을 사용해 사전 분포와 위의 표본 추출 모델을 결합하면 다음과 같습니다.

$$
p(\theta | y_1, ..., y_{129}) = \frac{y_1, ..., y_{129})p(\theta)}{p(y_1, ..., y_{129}} ( \rightarrow \ p(\theta) = 1)
$$

$$
= p(y_1, ..., y_{129}|\theta) \times \frac{1}{p(y_1, ..., y_{129})} ( \rightarrow \text{분모 부분은} \ \theta \text{에 의존하지 않음})
$$

$$
\propto p(y_1, ..., y_{129}|\theta)
$$

이 식의 의미하는 것은 사후 분포 $p(\theta|y_1, ..., y_{129})$가 $p(y_1, ..., y_{129}|\theta)$를 $\theta$에 의존하지 않는 어떠한 식으로 나눈 것과 같다는 것입니다(즉 각자 $\theta$에 대한 함수로서 서로 비례한다(proprotional)). 즉 이 두 가지 $\theta$에 대한 함수가 모양은 같으나 scale이 다르다는 것을 의미합니다.

### 데이터와 사후 분포

- 총 129명의 대상에게 설문조사 실시
- 118명의 사람이 행복하다고 응답(91%)
- 11명의 사람이 일반적으로 행복하지 않다고 응답{9%)

즉 $\theta$가 주어졌을 때 이 데이터들의 확률은 다음과 같습니다.

$$
p(y_1, ..., y_{129}|\theta) = \theta^{118}(1-\theta)^{11}
$$


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,6))
import numpy as np
from scipy.stats import binom, beta

# Panel 1

ax1 = plt.subplot(311)

k = 118 # 행복하다고 응답한 사람의 수
n = 129 # 전체 응답자 수

theta = np.linspace(0, 1, 100) # theta는 0부터 1까지 
sampling_model = binom.pmf(118, n, theta) 
plt.plot(theta, sampling_model ,color = 'grey')
plt.ylabel(r'$p(y_1, ... ,y_{129} | \theta)$')
plt.xlabel(r'$\theta$')




# Panel 2
  
ax2 = plt.subplot(313)

posterior = beta.pdf(theta, 1 + k, 1 + 129 - k)
plt.plot(theta, posterior, color = 'k', label = r'$p(\theta | y)$')
plt.yticks([0,5,10,15])
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta | y_1, ..., y_{129})$')
plt.axhline(1)

plt.show()
```


    

![output_17_0](https://user-images.githubusercontent.com/57588650/110318030-5531d400-8050-11eb-85c7-6da67199ef19.png)

    


**그림. 3.1.** 위의 그래프는 데이터의 표본 확률을, 아래 그래프는 사후 분포를 $\theta$에 대한 식으로 나타낸 것입니다. 여기서 주목할 점은 균등 분포로 표현된 사전 분포가(아래 그래프에서 하늘색 직선) 표본 확률에 비례하는 사후 분포를 산출해낸다는 것입니다.

위의 그래프에서 볼 수 있듯, 표본 확률과 사후 확률 $p(\theta|y_1, ..., y_{129})$의 모양이 같습니다. 즉 $\theta$의 실제 값은 0.91 근처, 최소한 0.8 이상일 확률이 매우 높습니다.

그러나 더 정확한 값을 구하기 위해서는 모양 뿐만 아니라 $p(\theta|y_1, ..., y_n)의 크기 또한 알아야 합니다. 베이즈 법칙에서 다음을 구할 수 있었습니다.

$$
p(\theta|y_1, ...,y_{129}) = \theta^{118} (1 - \theta)^11 \times p(\theta) / p(y_1, ..., y_{129})
$$

$$
= \theta^{118}(1 - \theta)^{11} \times 1 / p(y_1, ..., y_{129})
$$

그런데 스케일, 또는 "정규화 상수(normalizing constant)"라고 불리는 $1/p(y_1, ..., y_{129})$는 다음과 같은 적분을 통해 계산할 수 있음이 밝혀졌습니다.

$$
\int^1_0 \theta^{a-1}(1 - \theta)^{b -1} d \theta = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}.(\text{여기서} \ \Gamma (a) \ \text{는} (a-1)!) 
$$

계산)

- 조건1. $\int^1_0 p(\theta|y_1, ..., y_{129}) = 1$
- 조건2. $p(\theta|y_1, ...,y_{129}) = \theta^{118}(1 - \theta)^{11} \times 1 / p(y_1, ..., y_{129})$ (베이즈 법칙)

$$
1 = \int^1_0 p(\theta | y_1, ..., y_{129}) d\theta \ (\text{조건 1 사용})
$$

$$
1 = \int^1_0 \theta^{118}(1 - \theta)^{11} / p(y_1, ..., y_{129}) d\theta (\text{조건 2 사용})
$$

$$
1 = \frac{1}{p(y_1, ..., y_{129})} \int^1_0 \theta^{118}(1-\theta)^{11} d\theta
$$

$$
1 = \frac{1}{p(y_1, ..., y_{129})} \frac{\Gamma(119) \Gamma(12)}{\Gamma(131)} \ \ \ \text{(위의 계산 결과를 사용)}
$$

즉 

$$
p(y_1, ..., y_{129}) = \frac{\Gamma(119) \Gamma(12)}{\Gamma(131)}
$$

라는 결과가 나오게 되고 이것은 118개의 1이 있고 11개의 0이 있는 수열 {$y_1, ..., y_{129}$} 모두에 적용됩니다. 자 이제 모두를 합쳐봅시다

$$
p(\theta|y_1, ..., y_{129}) = \frac{\Gamma(131)}{\Gamma(119) \Gamma(12)} \theta^{118}(1 - \theta)^{11}
$$

$$
= \frac{\Gamma(131)}{\Gamma(119) \Gamma(12)} \theta^{119 -1}(1-\theta)^{12-1}.
$$

익숙한 식 아닌가요? 바로 모수가 $\alpha = 119, \beta = 12$인 베타 분포입니다. 위의 그래프를 그릴 때 알 수 있듯, 베타 분포는 파이썬에서 scipy.stats.beta.pmf($\theta$, a, b)를 통해 계산할 수 있습니다.

### 베타 분포

0과 1 사이의 미지의 수 $\theta$는 다음과 같은 $beta(a,b)$ 분포를 가지게 됩니다.

$$
p(\theta) = \text{scipy.stats.beta.pmf}(\theta, a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \theta^{a-1}(1-\theta)^{b-1} \ \ \text{for} \ 0 \leq \theta \leq 1.
$$

mode = $\frac{a-1}{(a-1) + (b-1)}$ if a > 1 & b > 1;

E($\theta$) = $\frac{a}{a+b}$

Var($\theta$) = $\frac{ab}{(a+b+1)(a+b)^2} = E[\theta] \times \frac{E[1-\theta]}{a + b + 1}$

### 3.1.1 교환 가능한 이항(binary) 데이터 추론

### *Uniform 사전 분포 하에서의 사후분포 추론*

만약

$$
Y_1, ..., Y_n | \theta \sim^{\text{i.i.d.}} \ \text{binary}(\theta)
$$

라면 다음과 같은 식이 성립함을 위에서 증명했습니다.

$$
p(\theta|y_1, ..., y_n) = \theta^{\Sigma y_1} (1-\theta)^{n - \Sigma y_i} \times p(y_1, . . ., y_n).
$$

그렇다면, 어떠한 두 $\theta$값 $\theta_a, \theta_b$일 때의 확률의 비율을 구해보면 다음과 같습니다.

$$
\frac{p(\theta_a|y_1, ..., y_n)}{p(\theta_b|y_1, ..., y_n)} = \frac{\theta_a^{\Sigma y_1} (1-\theta_a)^{n - \Sigma y_i} \times p(y_1, . . ., y_n)}{\theta_b^{\Sigma y_1} (1-\theta_b)^{n - \Sigma y_i} \times p(y_1, . . ., y_n)}
$$

$$
= \bigg ( \frac{\theta_a}{\theta_b} \bigg)^{\Sigma y_i}  \bigg(\frac{1-\theta_a}{1-\theta_b} \bigg)^{n - \Sigma y_i} \frac{\theta_a}{\theta_b} 
$$

즉, $\theta_b$에 대한 $\theta_a$의 상대 확률 밀도는 오직 $\sum^n_{i=1}y_i$를 통해서만 $y_1, ..., y_n$에 의존합니다. 이는 $y_i$가 어떤 순서로 나열되어 있든지 모든 $y_1, ..., y_n$의 합을 통해 데이터 $y_1, ..., y_n$이 주어졌을 때 두 $\theta$값을 가질 확률의 비율을 구할 수 있다는 뜻입니다. 따라서 이 상황에서 $y_1, ..., y_n$은 교환가능합니다. 즉 다음과 같이 쓸 수 있습니다.

$$
Pr(\theta \in A | Y_1 = y_1, ... , Y_n = y_n) = Pr \bigg( \theta \in A | \sum^n_{i=1}Y_i = \sum^n_{i=1}y_i \bigg)
$$

이것이 의미하는 것은, $\sum Y_i$만으로도 충분히 $\theta$와 $p(y_1, ..., y_n | \theta)를 추론할 수 있기 때문에 이것이 '충분 통계량(sufficient statistic)'이란 것입니다. 이 상황에서는

$$
Y_1, ..., Y_n | \theta \sim^{\text{i.i.d.}} \text{binary}(\theta)
$$

이기 때문에, 충분 통계량 $Y = \Sigma^n_{i=1}Y_i$는 파라미터가 ($n, \theta$)인 이항 분포(binomial distribution)입니다.


### *이항 분포*

확률변수 $Y \in$ {0,1,..,n}은 다음과 같은 pmf를 가질 때 binomial($n, \theta$) 분포를 가진다고 합니다.

$$
Pr(Y=y|\theta) = \text{scipy.stats.binom.pmf}(y,n,\theta) = \binom n y \theta^y(1-\theta)^{n-y}, y \in (0,1,...,n)
$$


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,7))
import numpy as np
from scipy.stats import binom

# n = 10, theta = 0.2

ax1 = plt.subplot(221)

x1 = np.arange(0,11,1)

y1 = binom.pmf(x1, 10, 0.2)

ax1.bar(x1, y1, width = 0.1)
plt.ylabel(r'$p(Y=y|\theta = 0.2, n=10)$')
plt.xlabel('y')
plt.ylim(-0.01,0.31)


# n = 10, theta = 0.8

ax2 = plt.subplot(222)

x1 = np.arange(0,11,1)

y1 = binom.pmf(x1, 10, 0.8)

ax2.bar(x1, y1, width = 0.1)
plt.ylabel(r'$p(Y=y|\theta = 0.8, n=10)$')
plt.xlabel('y')
plt.ylim(-0.01,0.31)

# n = 100, theta = 0.2


ax2 = plt.subplot(223)

x2 = np.arange(0,101,1)

y2 = binom.pmf(x2, 100, 0.2)

ax2.bar(x2, y2, width = 0.6)
plt.ylabel(r'$p(Y=y|n = 100, \theta = 0.2)$')
plt.xlabel('y')
plt.ylim(-0.001, 0.11)

# n = 100, theta = 0.8

ax2 = plt.subplot(224)

x2 = np.arange(0,101,1)

y2 = binom.pmf(x2, 100, 0.8)

ax2.bar(x2, y2, width = 0.6)
plt.ylabel(r'$p(Y=y|n = 100, \theta = 0.8)$')
plt.xlabel('y')
plt.ylim(-0.001, 0.11)
plt.show();
```


    

![output_41_0](https://user-images.githubusercontent.com/57588650/110318034-55ca6a80-8050-11eb-913a-6ffc56668346.png)

    


**그림 3.2. & 3.3.** n=10, 100이고 $\theta \in$ {0.2, 0.8} 일 때의 이항분포 그래프들

$\theta$가 주어졌을 때, 이항분포의 통계량 : 

$$
E[Y|\theta] = n\theta
$$

$$
Var[Y|\theta] = n\theta(1-\theta)
$$

**사후분포**

데이터 $Y=y$가 관찰됐을 때 $\theta$의 사후 분포는 다음과 같이 구할 수 있습니다.

$$
p(\theta|y) = \frac{p(y|\theta)p(\theta)}{p(y)}
$$

$$
=\frac{\binom n y \theta^y(1-\theta)^{n-y}p(\theta)}{p(y)}
$$

$$
= c(y)\theta(y)(1-\theta)^{n-y}p(\theta) \ \ , \ \ (c(y) = \theta\text{와 관련 없는} y \text{에 관한 함수})
$$

이 때, 위에서 배운 방법으로 정규화 상수(normalizing constant) $c(y)$를 다음과 같이 구할 수 있습니다.

$$
1 = \int^1_0 c(y) \theta^y (1-\theta)^{n-y} d\theta
$$

$$
1 = c(y) \int^1_0 \theta^y (1-\theta)^{n-y} d\theta
$$

$$
1 = c(y) \frac{\Gamma(y+1)\Gamma(n-y+1)}{\Gamma(n+2)}.
$$

$$
\therefore c(y) = \frac{\Gamma{n+2}}{\Gamma(y+1)\Gamma(n-y+1)}
$$

이를 위의 사후분포 식에 대입하면 다음과 같은 결과를 얻을 수 있습니다.

$$
p(\theta|y) = \frac{\Gamma(n+2)}{\Gamma(y+1)\Gamma(n - y + 1)} \theta^y(1-\theta)^{n-y}
$$

$$
= \frac{\Gamma(n+2)}{\Gamma(y+1)\Gamma(n - y + 1)} \theta^{(y+1)-1} (1-\theta)^{(n-y+1) - 1}
$$

$$
= beta(y+1, n-y+1)
$$

행복 데이터를 예시에서, $Y \equiv \Sigma Y_i = 118$이고

$$
n = 129, Y \equiv \Sigma Y_i = 118 \Rightarrow \theta|(Y = 118) \sim beta(119,12)
$$

임을 확인했습니다.

즉 $\Sigma y_i = y = 118$일 때, $p(\theta|y_1, ..., y_n) = p(\theta|y) = beta(119,12)$임을 보이는 것이 이 모델과 사전 분포의 결과물로 충분합니다.

다른 말로 표현하자면, {$Y_1 = y_1, ..., Y_n = y_n$}이 포함하고 있는 정보는 $Y = \Sigma Y_i, y = \Sigma y_i$일 때 {$Y=y$}가 포함하고 있는 정보와 같습니다.

**베타 사전 분포 하에서의 사후 분포**

$p(\theta) = 1$인 균등 분포는 다음과 같이 $a=1, b=1$인 베타 분포를 따른다고 볼 수 있습니다.

$$
p(\theta) = \frac{\Gamma(2)}{\Gamma(1)\Gamma(1)} \theta^{1-1} (1 - \theta)^{1-1} = \frac{1}{1 \times 1} = 1
$$

이전 단락(사후 분포)에서 다음과 같은 결과를 얻었습니다.

$$
\text{만약} \ \begin{cases} \theta \sim beta(1,1) \ (\text{uniform}) \newline Y \sim \text{binomial}(n, \theta) \end{cases} \text{라면}, (\theta|Y=y) \sim beta(1+y, 1+n-y)
$$

즉 우리의 사전 분포가 beta(a = 1, b = 1)일 때, 사후 분포를 구하기 위해서는 파라미터 $a$에 1을 더하고, $b$에 {$y_1, ..., y_n$}중 0의 갯수를 더하기만 하면 됩니다.

이제 이 방법이 임의의 베타 분포에도 적용되는지 봅시다.

$$
\theta \sim beta(a,b)
$$

$$
Y|\theta \sim binomial(n,\theta)
$$

일 때,

$$
p(\theta|y) = \frac{p(\theta)p(y|\theta)}{p(y)}
$$

$$
= \frac{1}{p(y)} \times \underbrace{\frac{\Gamma(a+b)}{\Gamma(a) \Gamma(b)} \theta^{a-1}(1-\theta)^{b-1}}_{p(\theta)} \times \underbrace{\binom n y \theta^y (1-\theta)^{n-y}}_{p(y|\theta)}
$$

$$
= c(n,y,a,b) \times \theta^{a + y - 1} (1-\theta)^{b+n-y-1} \ , \ \ (c(n,y,a,b) = n,y,a,b \text{에 의존하는 정규화 상수})
$$

$$
= \text{scipy.stats.beta.pdf}(\theta, a+y, b+n-y)
$$

라고 구할 수 있습니다.

이 연산에서 가장 중요한 것은 마지막 두 라인입니다. 밑에서 두 번째 라인에서 볼 수 있듯이 $p(\theta|y)$가 $\theta$에 대한 함수이기 때문에 c(n,y,a,b)는 상수로 취급되고, 이것은 $p(\theta|y)$가 $\theta^{a + y - 1} (1-\theta)^{b+n-y-1}$와 비례한다는 것을 의미합니다. 즉 scipy.stats.beta.pdf($\theta, a+y, b+n-y$)와 같은 모양이라는 뜻이죠. 

또한 $p(\theta|y)$와 scipy.stats.beta.pdf($\theta, a+y, b+n-y$)가 모두 적분하면 1이므로 스케일 또한 같습니다. 즉 사실은 $p(\theta|y)$와 scipy.stats.beta.pdf($\theta, a+y, b+n-y$)는 같은 함수인 것입니다. 

앞으로 계속해서 이를 활용해 우리는 사후 분포가 어떠한 알려진 확률 밀도 함수와 비례하고, 따라서 그 확률 밀도 함수와 반드시 같다는 것을 찾아내볼 것입니다. 

**켤레성(Conjugacy)**

지금까지 배운 "베타 사전 분포와 이항 표본 모델이 결합되어 베타 사후 분포를 만드는 것"을 베타 사전 분포의 클래스가 이항 표본 모델의 "켤레(congutate)"라고 부릅니다.

### 정의 4 (켤레(Conjugate))

만약 다음이 성립한다면, $\theta$의 사전 분포의 클래스 $\mathcal{P}$를 표본 모델 $p(y|\theta)$의 "켤레(conjugate)"라고 부릅니다.

$$
p(\theta) \in \mathcal{P} \Rightarrow p(\theta|y) \in \mathcal{P}
$$

켤레 사전 분포(conjugate prior)를 이용한다면, 계산을 쉽게 할 수는 있지만 우리의 실제 사전 정보를 반영하긴 쉽지 않습니다. 그러나 켤레 사전 분포들을 결합한다면, 아주 유연하고 계산을 추적 가능(computationally tractable)하게 만들 수 있습니다.(예제 3.4와 3.5를 보세요)

**정보들을 결합하기**

만약 $\theta|${Y=y} $\sim beta(a+y, b+n-y)$라면

$E[\theta|y] = \frac{a + y}{a + b + n}$, mode[$\theta|y$] = $\frac{a + y - 1}{a + b + n - 2}$, Var[$\theta|y$] = $\frac{E[\theta|y]E[1-\theta|y]}{a + b + n + 1}$

입니다.


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,10))
import numpy as np
from scipy.stats import beta

# Panel 1

ax1 = plt.subplot(221)

x1 = np.arange(0,1,0.01)

y0 = beta.pdf(x1, 1, 1) # beta(1,1) prior distribution

y1 = beta.pdf(x1, 1 + 1, 1 + 5 - 1) #p(theta|y) = beta(a+y, b + n - y)

ax1.plot(x1, y0,  color = 'grey', label = 'prior') # prior distiribution
ax1.plot(x1, y1, label = 'posterior') # posterior distribution

plt.title(r'beta(1,1) prior distiribution & data with n = 5 ,$\Sigma y_i = 1$')
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel('y')
plt.ylim(-0.01,3)
plt.legend()


# Panel 2

ax2 = plt.subplot(222)

x1 = np.arange(0,1,0.01)

y0 = beta.pdf(x1, 3, 2) # beta(1,1) prior distribution

y1 = beta.pdf(x1, 3 + 1, 2 + 5 - 1) #p(theta|y) = beta(a+y, b + n - y)

ax2.plot(x1, y0,  color = 'grey', label = 'prior') # prior distiribution
ax2.plot(x1, y1, label = 'posterior') # posterior distribution

plt.title(r'beta(3,2) prior distiribution & data with n = 5 ,$\Sigma y_i = 1$')
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel('y')
plt.ylim(-0.01,3)
plt.legend()

# Panel 3

ax3 = plt.subplot(223)

x1 = np.arange(0,1,0.01)

y0 = beta.pdf(x1, 1, 1) # beta(1,1) prior distribution

y1 = beta.pdf(x1, 1 + 20, 1 + 100 - 20) #p(theta|y) = beta(a+y, b + n - y)

ax3.plot(x1, y0,  color = 'grey', label = 'prior') # prior distiribution
ax3.plot(x1, y1, label = 'posterior') # posterior distribution

plt.title(r'beta(1,1) prior distiribution & data with n = 100 ,$\Sigma y_i = 20$')
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel('y')
plt.ylim(-0.5,10.5)
plt.legend()

# Panel 4

ax4 = plt.subplot(224)

x1 = np.arange(0,1,0.01)

y0 = beta.pdf(x1, 3, 2) # beta(1,1) prior distribution

y1 = beta.pdf(x1, 3 + 20, 2 + 100 - 20) #p(theta|y) = beta(a+y, b + n - y)

ax4.plot(x1, y0,  color = 'grey', label = 'prior') # prior distiribution
ax4.plot(x1, y1, label = 'posterior') # posterior distribution

plt.title(r'beta(3,2) prior distiribution & data with n = 100 ,$\Sigma y_i = 20$')
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel('y')
plt.ylim(-0.5,10.5)
plt.legend()


plt.show();
```


    

![output_63_0](https://user-images.githubusercontent.com/57588650/110318035-56630100-8050-11eb-9c4d-3866aa1917a1.png)

    


**그림 3.4.** 두 개의 다른 표본의 크기와 두 개의 다른 사전 분포를 가진 베타 사후 분포의 그래프들입니다. 왼쪽과 오른쪽을 비교해보면서 사전 분포가 주는 사후 분포로의 영향을 확인하고, 위 아래를 비교하면서 표본 크기의 사후 분포에 대한 영향을 확인해보세요

사후 기댓값 $E[\theta|y]$는 다음과 같이 사전 분포와 데이터의 정보를 결합하여 쉽게 구할 수 있습니다.

$$
E[\theta|y] = \frac{a+y}{a+b+n}
$$

$$
= \frac{a+b}{a+b+n}\frac{a}{a+b} + \frac{n}{a+b+n}\frac{y}{n}
$$

$$
= \frac{a + b}{a + b + n} \times \text{prior expectation} + \frac{n}{a + b + n} \times \text{data average.}
$$



이것이 의미하는 바는 바로 사후 기댓값(사후 평균)이 사전 기댓값과 표본 평균의 각각 a+b와 n에 비례하는 가중 평균이라는 것입니다. 즉 a와 b를 "사전 데이터"라고 해석할 수 있다는 것이죠:

$$
a \approx \text{"사전에 알려진 1의 갯수"}
$$

$$
b \approx \text{"사전에 알려진 0의 갯수"}
$$

$$
a+b \approx \text{"사전에 알려진 표본의 크기"}
$$



한가지 더 알 수 있는 것은, 만약 n이 a+b보다 크다면 데이터의 평균이 사전 기댓값보다 사후 분포에 더 많은 정보를 준다는 것입니다. 만약 n이 a+b보다 상당히 크다면

$$
\frac{a+b}{a+b+n} \approx 0, \ \ E[\theta|y] \approx \frac{y}{n}, \ \ Var[\theta|y] \approx \frac{1}{n} \frac{y}{n} \bigg (1-\frac{y}{n} \bigg). \ \ \ \ (n \rightarrow \infty \text{라고 생각해보세요})
$$

이 되고, 이는 사전 분포가 사후 분포에 아무런 영향도 주지 못한다는 것을 의미합니다.

**예측**

베이지안 추론의 가장 중요한 특징은 새로운 관찰값을 반영한 예측 분포(predictive distribution)가 존재한다는 것입니다. 
다음과 같은 binary 데이터를 예로 들어봅시다.

$$
y_1, ..., y_n = \text{n개의 binary 확률 변수}
$$

여기에 아직 관찰되지 않은 추가적인 데이터 $\tilde{Y} \in$ {0,1}이 같은 모집단으로부터 산출되었다고 합시다. 그렇다면 $\tilde{Y}$의 예측 분포(predictive distribution)은 다음과 같은 조건부 분포입니다.

$$
\tilde{Y} | Y_1 = y_1, ..., Y_n = y_n
$$

이것이 조건부 i.i.d.인 binary 확률 변수라면 추가적인 데이터 $\tilde{Y}$가 1 또는 0의 값을 가질 확률을 $\tilde{Y}|\theta$와 $\theta$의 사후 분포를 통해 구할 수 있습니다.

$$
Pr(\tilde{Y} = 1|y_1, ..., y_n) = \int Pr(\tilde{Y} = 1, \theta | y_1, ..., y_n) d\theta \ \ (\text{by rule of marginal probability})
$$

$$
= \int \underbrace{Pr(\tilde{Y} = 1 | \theta, y_1, ..., y_n)}_{= \theta, \ \theta\text{가 1이 나올 확률이므로,} \theta \text{도 주어졌다면} \tilde{Y} \text{도 1이 나올 확률은} \theta \text{이다.}  } p(\theta|y_1, ..., y_n) d\theta \ (\text{by rule of marginal probability})
$$

$$
= \int \theta p(\theta|y_1, ..., y_n) d\theta
$$

이것은 기댓값을 구하는 식과 같다. 따라서

$$
= E[\theta|y_1, ..., y_n] = \frac{a + \sum^n_{i=1}y_i}{a + b + n}
$$

이고, 같은 방식으로 $\tilde{Y} = 0$일 확률은 다음과 같다.

$$
Pr(\tilde{Y} = 0|y_1, ..., y_n) = 1 \ - \ E[\theta|y_1, ..., y_n] = \frac{b + \sum^n_{i-1}(1-y_i)}{a+b+n}
$$

이것을 통해 배울 수 있는 예측 분포에 대한 두 가지 중요한 포인트는 다음과 같습니다.

1. 예측 분포는 아직 알려지지 않은 값에는 의존하지 않습니다. 만약 의존한다면, 예측을 위해 이 분포를 사용할 수 없습니다.

2. 예측 분포는 관찰된 데이터에 의존합니다. 이 분포에서는 $\tilde{Y}$가 $Y_1, ..., Y_n$에 독립이 아닙니다.(2.7 섹센을 다시 보세요) 왜나하면 $Y_1, ..., Y_n$이 $\theta$에 대한 정보를 주고, 또 이것이 $\tilde{Y}$에 대한 정보를 주기 때문이죠. 만약 $\tilde{Y}$가 관찰된 데이터에 독립이라면 지금까지 관찰된 표본으로는 아직 추출되지 않은 모집단에 대해 아무것도 추론할 수 없습니다.


**예시**

균등 사전 분포(또는 beta(1,1))는 사전 데이터가 하나의 1과 하나의 0을 가지고 있다는 정보를 가지고있다는 것과 동일합니다. 이 사전분포 하에서는 $Y = \Sigma^n_{i=1}Y_i$일 때

$$
Pr(\tilde{Y} = 1 | Y = y) = E[\theta|Y=y] = \frac{2}{2+n} \frac{1}{2} + \frac{n}{2+n}\frac{y}{n},
$$

$$
mode(\theta|Y=y) = \frac{y}{n}
$$

입니다. 

왜 이 사후 기댓값과 사후 최빈값(mode)이 다른지 이해가시나요? 예를 들어 데이터가 모두 0인 경우($Y = \sum Y_i = 0$)는 mode($\theta|Y=0$) = 0이지만, $Pr(\tilde{Y} = 1| Y= 0) = \frac{1}{2+n}$입니다. 왜냐하면 사전 분포인 균등 사전 분포(또는 beta(1,1))가 0,1이 나올 확률이 같다(즉 각 1/2이다)는 정보를 포함하고 있기 때문입니다.

### 3.1.2 신뢰 영역(Confidence Regions)

파라미터의 실제 값이 들어있을만한 파리미터 공간의 영역을 찾기 위해, $Y=y$가 관측된 후

$$
Pr[l(y) < \theta < u(y)]
$$

가 큰 구간 $[l(y), u(y)]$를 구하고 싶습니다.

**정의 5**(베이지안 신뢰구간(Bayesian Coveraage))

$Y=y$가 관측됐을 때, 구간 $[l(y), u(y)]$는 다음과 같은 경우에 $\theta$에 대해 95% 베이지안 신뢰구간을 가진다고 합니다.

$$
Pr(l(y) < \theta < u(y)|Y=y) = .95
$$

이 구간이 뜻하는 바는 "데이터 $Y=y$를 관측한 후" 파라미터 $\theta$의 참값이 어디에 위치할지에 대한 정보입니다. 프리퀀티스트들은 이와는 다르게 데이터가 관측되기 전에 그 구간이 참값을 커버할 확률이라고 말합니다.

**정의 6**(프리퀀티스트 신뢰구간(frequentist coverage))

"데이터를 수집하기 전"에, **랜덤** 구간 $[l(y), u(y)]$는 다음과 같은 경우에 $\theta$에 대해 95% 프리퀀티스트 신뢰구간을 가진다고 합니다.

$$
Pr(l(y) < \theta < u(y)|\theta) = .95.
$$

하지만, 데이터를 관찰하기 전에 수행되는 프리퀀티스트의 신뢰구간은 데이터를 관찰한 이후에 다음과 같이 된다는 문제가 있습니다.

$$
Pr(l(y) < \theta < u(y)|\theta) = \begin{cases} 0 \ \ if \ \theta \notin [l(y), u(y)]; \newline 1 \ \ if \ \theta \in [l(y), u(y)]. \end{cases}
$$

즉 0 또는 1의 값 밖에 가지지 못한다는 점이죠. 그렇다고 해서 프리퀀티스트 신뢰구간이 쓸모 없는 것은 아닙니다. 예를 들어 각자 그들 자신의 신뢰구간을 형성하는 수많은 서로 관련 없는 실험을 수행한다고 가정합시다. 만약 각각의 구간이 95% 프리퀀티스트 신뢰구간을 가진다면, 95%의 확률로 그 구간이 올바른 파라미터 값을 포함한다고 예측할 수 있습니다.

 그렇다면 베이지안 신뢰구간과 프리퀀티스트 신뢰구간은 어떤 관계가 있을까요? Hartigan(1966)은 베이지안 신뢰구간이 추가적으로 다음과 같은 특성이 있다는 것을 보였습니다.
 
 $$
 Pr(l(Y) < \theta < u(Y)|\theta) = .95 + \epsilon_n \ \ (|\epsilon_n| < \frac{a}{n}, a = \text{어떠한 상수})
 $$

즉 베이지안 신뢰구간 역시 점근적으로(n이 커질 수록) 95% 프리퀀티스트 신뢰구간에 근사합니다. 즉 베이지안이든 베이지안이 아니든 점근적으로 같은 신뢰구간을 가지게 됩니다. 추가적인 베이지안과 베이지안이 아닌 방법론으로 만들어진 신뢰구간 사이의 비슷함은 Severini(1991), Sweeting(2001)의 논문을 참고해봅시다.

**분위수 기반 구간(Quantile-based interval)**

신뢰구간을 구하는 가장 쉬운 방법은 사후 분위수를 사용하는 것입니다. 

$100 \times (1-\alpha)$% 분위수 기반 신뢰 구간을 구하는 방법은 다음을 만족하는 $\theta_{\alpha/2} < \theta_{1-\alpha/2}$를 구하는 것입니다.

1. $Pr(\theta < \theta_{\alpha/2} | Y = y) = \alpha/2;$
2. $Pr(\theta > \theta_{1 - \alpha/2}|Y=y) = \alpha/2;$

여기서 구할 수 있는 $\theta_{\alpha/2}, \theta_{1-\alpha/2}$는 각각 $\theta$의 $\alpha, 1-\alpha/2$ 사후 분위수입니다. 이들은 또한 다음을 만족합니다.

$$
Pr(\theta \in [\theta_{\alpha/2}, \theta_{1-\alpha/2}]|Y=y) = 1 - Pr(\theta \notin [\theta_{\alpha/2}, \theta_{1- \alpha/2}]|Y=y)
$$

$$
= 1 - [\underbrace{Pr(\theta < \theta_{\alpha/2} | Y=y)}_{\alpha/2} + \underbrace{Pr(\theta > \theta_{1-\alpha/2}|Y=y)}_{\alpha/2}]
$$

$$
= 1- \alpha
$$

**예시 : 이항 샘플과 균등 사전 분포**

n = 10의 이진 확률 변수에서 조건부 독립인 표본 추출을 시행해 2번의 1이 나왔다(Y = $\sum y_i$= 2)고 가정합시다. 지금까지 배운 방식으로 사후 분포를 구하면 다음과 같습니다.

$$
\theta|Y=y \ \sim \ \text{beta}( 1 + 2, 1 + 8 )
$$

여기에서 95% 베이지안 신뢰구간을 구한다고 하면, 이 베타 분포의 .025와 .975 분위수를 통해 구할 수 있을 것입니다. 이 분위수들은 각각 0.06, 0.52이고, 즉 $\theta \in [0.06, 0.52]$일 사후 확률은 95%입니다.


```python
from scipy.stats import beta

# prior
a = 1
b = 1

# data

n = 10
y = 2

# posterior quantile

beta.ppf([0.025, 0.975], a + y, b + n - y)
```




    array([0.06021773, 0.51775585])



### 최고 사후 밀도(Highest Posterior Density, HPD) 구간

**그림 3.5**는 이전 예시에서의 사후 분포와 $\theta$에 대한 95% 신뢰구간을 보여줍니다. 여기에서 주목할 것은 신뢰구간 밖에 신뢰구간 안에 있는 지점보다 높은 확률(밀도)를 가지는 곳이 있다는 것입니다. 이런 곳이 없도록 더 제한된 형태의 구간이 바로 HPD 구간입니다.


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,6))
import numpy as np
from scipy.stats import beta

# Posterior Distribution

x1 = np.arange(0,1,0.01)
posterior = beta.pdf(x1, 3, 9)

plt.plot(x1, posterior)
plt.ylabel(r'$p(\theta|y)$')
plt.xlabel(r'$\theta$')

# quantile based confidence interval

q1, q2 = beta.ppf([0.025, 0.975], 3, 9)

plt.axvline(q1, color = 'grey')
plt.axvline(q2, color = 'grey')


# 신뢰구간 밖에 신뢰구간 안에 있는 지점보다 높은 확률(밀도)를 가지는 곳의 예시

## 신뢰구간 밖에 있는 점
x = 0.055
y = beta.pdf(x, 3,9)

plt.scatter(x,y, color = 'red')

## 신뢰구간 안에 있는 점
x = 0.5
y = beta.pdf(x,3,9)

plt.scatter(x,y,color = 'green')

plt.show()
```


    

![output_88_0](https://user-images.githubusercontent.com/57588650/110318038-56fb9780-8050-11eb-9d29-bc4d29548fc4.png)


    


**그림 3.5.** 베타 사후 분포와 수직선으로 표현된 95% 신뢰구간입니다. 여기서 볼 수 있듯, 신뢰구간 밖에 있는 빨간 점의 확률이 신뢰구간 안에 있는 초록 점보다 높은 곳에 있습니다. 이런 경우를 제한한 신뢰구간이 바로 HPB 영역입니다.

**정의 7(HPD 영역)**

파라미터 공간의 부분집합 $s(y) \subset \Theta$을 포함하는 $100 \times (1-\alpha)$% HPD 영역은 다음을 만족합니다. 

1. $Pr(\theta \in s(y) | Y = y) = 1 - \alpha$

2. 만약 $\theta_{\alpha} \in s(y)$이고, $\theta_b \notin s(y)$라면, $p(\theta_a | Y= y) > p(\theta_b|Y=y)$이다. 즉, 신뢰구간 안에 포함된 확률(밀도)은 포함되지 않은 확률(밀도)보다 항상 큽니다.

제약에 따라 HPD 영역 안에 있는 포인트들은 항상 영역 밖에 있는 포인트들보다 높은 확률을 가집니다. 그러나 사후 밀도 함수가 multimodal(두 개의 꼭대기가 있는)이라면, HPD 영역은 구간으로 표현되지 않을 것입니다. **그림 3.6.** 에 HPD를 구하는 기본적인 개념이 담겨있습니다. 

1. 수평선을 꼭대기에서 부터 점차 내려가면서, 그 아래의 영역이 $(1-\alpha)$가 될 때 멈추고 

2. 해당하는 $\theta$값들을 구합니다. 

이렇게 하면 모든 신뢰 구간 안에서 확률이 수평선보다 높게 됩니다. 예를 들어 위의 이항 분포 예시에서 HPD 영역은 [0.04, 0.048]입니다. 이것은 분위수 기반 구간보다 더 좁지만(또는 더 정확하지만) 둘 모두 95%의 사후 확률을 포함합니다. 

![IMG_0FA9D025221B-1](https://user-images.githubusercontent.com/57588650/110317294-4991dd80-804f-11eb-9961-178d0a52275f.jpeg)

**그림. 3.6.** 확률에 따라 달라지는 HPD를 보여주는 그래프입니다. 점선으로 표현된 것은 95% 분위수 기반 구간입니다.


