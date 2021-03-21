---
title: A First Course in Bayesian Statistical Methods Chapter 5. Multi Parameter Model
date: 2021-03-18T17:34:01+09:00
categories: ["A First Course In Bayesian Statistical Methods"]
tags: ["Bayesian", "Python"]
---

# A First Course in Bayesian Statistical Methods

## 5. 파라미터가 여러개인 정규 모델(Multi parameter Normal Model)

### 5.1 'nuisance 파라미터'로 평균화

결합(joint) 그리고 주변(marginal) 사후 분포의 아이디어를 수학적으로 표현하기 위해서, 두 가지 가정을 해보겠습니다.

1. 파라미터는 $\theta = (\theta_1, \theta_2)$와 같이 두 파트를 가지고 있습니다.
2. 두 개 중 $\theta_1$의 추론에만 관심이 있습니다. 이 때 관심 없는 $\theta_2$를 'nuiance(=성가신, 귀찮은) 파라미터'라고 합니다.

정규 분포를 예로 들면,

$$
y|\mu,\sigma^2 \sim N(\mu, \sigma^2)
$$

에서, 각각의 $\mu(=\theta_1)$와 $\sigma^2(=\theta_2)$는 미지수이고, 보통은 $\mu$에 관심이 있습니다.

여기에서 관찰된 데이터가 주어졌을 때, 관심있는 파라미터의 조건부 분포(이 상황에서는 $p(\theta_1|y)$를 찾고싶습니다. 그것은 다음과 같은 결합 사후 밀도 함수를

$$
p(\theta_1, \theta_2 | y) \propto p(y|\theta_1, \theta_2)p(\theta_1, \theta_2)
$$

$\theta_2$에 대해 평균화함으로써 얻을 수 있습니다.

$$
p(\theta_1|y) = \int p(\theta_1,\theta_2|y)d\theta_2
$$

이것을 인수분해를 통해 다음과 같이 표현할 수도 있습니다.

\begin{align}
p(\theta_1|y) = \int p(\theta_1,\theta_2|y)d\theta_2 \newline
= \int \frac{p(\theta_1, \theta_2, y)}{p(y)} d\theta_2 \newline
= \int p(\theta_1|\theta_2,y)p(\theta_2,y)\frac{1}{p(y)} d\theta_2 \newline
= \int p(\theta_1|\theta_2,y)p(\theta_2|y)d\theta_2 \ \ \ \ \ \ \ \ \ \ \ \ \ \ (5.1)
\end{align}

이것은 관심있는 $\theta_1$의 사후 분포 $p(\theta_1|y)$가 nuisance 파라미터 $\theta_2$가 주어졌을 때의 조건부 사후 분포 $p(\theta_1|\theta_2,y)$를 $\theta_2$가 가질 수 있는 서로 다른 값들($p(\theta_2|y)$)을 가중치로 해서 결합한 것이란걸 보여줍니다.

가중치는 $\theta_2$의 사후 밀도, 즉 $\theta_2$의 사전 분포와 데이터로부터 수집한 증거의 결합에 의존합니다. 이러한 'nuiance' 파라미터로 평균화 하는 과정은 일반적으로(generally) 해석될 수 있습니다. 예를 들어 $\theta_2$는 위에서 본 연속 분포에서의 분산이 아니라 각각의 모델을 나타내는 이산적인(discrete) 요소가 될 수도 있습니다.

우리가 실제로 저 적분을 계산하지는 않지만, 이것은 파라미터가 여러개인 모델을 만들고 계산하는데 중요한 실용적인 전략을 제공합니다. 그 과정은 다음과 같습니다.

1. $\theta_2$의 주변 사후 분포로부터 $\theta_2$를 뽑습니다.
2. 뽑힌 $\theta_2$를 사용해 조건부 사후 분포 $p(\theta_1|\theta_2,y)$에서 $\theta_1$을 뽑습니다.

이 방식을 사용해 위의 적분을 우회적으로 구할 수 있게 됩니다. 이러한 분석 방법의 정석적인 예시가 바로 미지의 평균과 분산을 가지고 있는 정규 모델입니다. 지금부터는 이것을 배워보도록 합시다!

### 5.2 noninformative 사전 분포를 가지는 정규 데이터

우선 다음과 같은 조건을 만족하는 일변수 정규 모델로 예시를 들어보겠습니다.

1. noninformative 사전 분포
2. 일변수 정규 모델 $N(\mu, \sigma^2$로 부터 온 n개의 독립된 y_1, ... , y_n

### noninformative 사전 분포

noninformative prior(=vague prior)는 앞에서 배운 uniform 분포, beta(1,1) 분포와 같이 사후 분포에 아무런 영향을 주지 못하는 사전분포를 말하는 것입니다. 이 예시에서 $\mu, \sigma)$에 대한 vague 사전 분포는 $\mu$(=location)와 $\sigma$(=scale) 파라미터의 사전 독립을 가정했을 때, $(\mu, log\sigma)$에서 균등 분포이고 다음과 같이 표현할 수 있습니다.(BDA 52p 참조)

$$
p(\mu, \sigma^2) = p(\mu) p(\sigma^2) \propto (\sigma^2)^{-1}
$$

### 결합 사후 분포, $p(\mu, \sigma^2|y)$

이러한 전통적인 improper 사전 밀도 함수($\sum = 1$이 아닌 사전 밀도 함수) 하에서, 결합 사후 분포는 likelihood 함수에 $1/\sigma^2$을 곱한 것에 비례합니다.

\begin{align}
p(\mu, \sigma^2|y) \propto \frac{1}{\sigma^2} \times \prod_{i=1}^n \frac{1}{\sigma}exp\bigg( -\frac{(y_i-\mu)^2}{2\sigma^2} \bigg) \newline
= \sigma^{-n-2}exp\bigg( -\frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \mu)^2 \bigg) \newline
= \sigma^{-n-2}exp\bigg( -\frac{1}{2\sigma^2} \bigg[\sum_{i=1}^n(y_i - \bar{y} + \bar{y} - \mu)^2 \bigg] \bigg) \newline
= \sigma^{-n-2} exp \bigg(-\frac{1}{2\sigma^2} \bigg[\sum_{i=1}^n(y_i-\bar{y})^2 + n(\bar{y} - \mu)^2 \bigg] \bigg) \newline
= \sigma^{-n-2} exp \bigg( -\frac{1}{2\sigma^2} \bigg[ (n-1) \underbrace{\frac{\sum_{i=1}^n(y_i - \bar{y})^2}{n-1}}_{= s^2(y_i \text{의 표본분산})} + n(\bar{y} - \mu)^2 \bigg] \bigg) \newline
= \sigma^{-n-2} exp \bigg( -\frac{1}{2\sigma^2} [(n-1)s^2 + n(\bar{y} - \mu)^2] \bigg) \ \ \ \ \ \ (5.2)
\end{align}

이 때 충분통계량은 $\bar{y}$와 $s^2$입니다.

### 조건부 사후 분포, $p(\mu|\sigma^2, y)$

(5.1)식과 같이 결합 사후 밀도를 인수분해 하기 위해서는, 첫 번째로 조건부 사후 밀도인 $p(\mu|\sigma^2, y)$를, 두 번째론 주변 사후 밀도인 $p(\sigma^2|y)$를 고려해야 합니다. $\mu|\sigma^2$의 사후 분포를 정하기 위해서는 4.2에서 배운 분산을 알 때의 정규분포 평균의 사후 분포를 구한 결과와 균등 사전 분포를 사용하면 됩니다.

$$
\mu|\sigma^2,y \sim N(\bar{y}, \sigma^2/n) \ \ \ \ \ \ (5.3)
$$

### 주변 사후 분포, $p(\sigma^2|y)$

$p(\sigma^2|y)$를 구하기 위해서는 (5.2)식을 $\mu$에 대하여 적분하면 됩니다.

$$
p(\sigma^2|y) \propto \int \sigma^{-n-2}exp \bigg(-\frac{1}{2\sigma^2}[(n-1)s^2 + n(\bar{y} - \mu)^2] \bigg)d\mu
$$

$$
= \sigma^{-n-2} exp\bigg(-\frac{1}{2\sigma^2}(n-1)s^2 \bigg) \int exp\bigg( -\frac{1}{2\sigma^2} n(\bar{y} - \mu)^2 \bigg) d\mu
$$

이 적분을 풀기 위해 

$$
exp\bigg(-\frac{1}{2\sigma^2}n(\bar{y}-\mu) \bigg)
$$

이 정규분포의 pdf와 비슷하게 생겼다는 아이디어를 떠올리면 됩니다. 즉

$$
exp\bigg(-\frac{1}{2\sigma^2}n(\bar{y}-\mu) \bigg) = exp\bigg(-\frac{(\bar{y} - \mu)^2}{2\frac{\sigma^2}{n}} \bigg)
$$

이고, 이것을 평균이 $\mu$이고 분산이 $\sigma^2/n$인 정규분포의 pdf라고 생각합시다. 그러면

$$
\int \frac{1}{\sqrt{2\pi\sigma^2/n}} exp\bigg(-\frac{(\bar{y} - \mu)^2}{2\frac{\sigma^2}{n}} \bigg) = 1
$$

$$
\therefore \int exp\bigg( -\frac{1}{2\sigma^2} n(\bar{y} - \mu)^2 \bigg) d\mu = \sqrt{2\pi\sigma^2/n}
$$

이고, 따라서 $p(\sigma^2|y)$는

$$
p(\sigma^2|y) \propto \sigma^{-n-2}  exp \bigg(-\frac{1}{2\sigma^2}(n-1)s^2 \bigg)\sqrt{2\pi\sigma^2/n}
$$

$$
\propto (\sigma^2)^{-(n+1)/2} exp \bigg( -\frac{(n-1)s^2}{2\sigma^2} \bigg) \ \ \ \ \ (5.4)
$$

입니다. 그리고 이것은 위에서 배운 다음과 같은 scaled inv-$\chi^2$분포의 pdf입니다.

$$
\sigma^2|y \sim \text{Inv-}\chi^2(n-1, s^2) \ \ \ \ \ \ (5.5)
$$

자 지금까지 결합 사후 분포 (3.2)를 조건부 사후 분포와 주변 사후 분포의 곱으로 인수분해해봤습니다.($p(\mu, \sigma^2|y) = p(\mu|\sigma^2,y)p(\sigma^2|y)$)

여기에서 신기한 것은, $\sigma^2$에 대한 주변 사후 분포가 표본 이론의 결과와 상당히 비슷하다는 것입니다. 여러분이 배웠듯이 $\sigma$(그리고 $\mu$)가 주어졌을 때 적절하게 스케일된 충분 통계량 $\frac{(n-1)s^2}{\sigma^2}$의 분포는 $\chi_{n-1}^2$입니다. 위의 pdf에서 exponential 안에 있는 식과 비슷하죠?

### 결합 사후 분포에서 표본 추출하기

결합 사후 분포에서 표본을 추출하는건 쉽습니다(R을 예로 들면 그냥 dbeta와 같은 함수를 쓰면 됩니다!): 처음으로 (5.5)에서 $\sigma^2$을 뽑고, 이것을 사용해 (3.3)에서 $\mu$를 뽑습니다. 

표본 추출을 통한 방법도 있지만, 이 예시에서의 사후 분포는 적분이 닫힌 형태로 풀리는 흔치 않은 파라미터가 여러개인 모델 중 하나입니다.

다른 방식으로 구해보면, 

### $\mu$에 대한 주변 사후 분포의 해석적인 형태(Analytic form)

모평균 $\mu$는 대부분의 연구에서 구하고자 하는 추정값입니다. 그렇기 때문에 베이지안 분석의 목적은 결합 사후 분포를 $\sigma^2$에 대해 적분한 $\mu$의 주변 사후 분포입니다. 
(5.1)에서 볼 수 있듯, 하나의 파라미터 $\mu$의 사후 분포는 정규분포와 scaled Inv-$\chi^2$의 결합입니다. 자 이제 결합 사후 확률 밀도 함수에서 $\sigma^2$에 대해 적분함으로서 $\mu$의 사후 확률 밀도 함수를 구해봅시다.

$$
p(\mu|y) = \int_0^{\infty} p(\mu, \sigma^2|y)d\sigma^2 = \int_0^{\infty} \sigma^{-n-2} exp \bigg( -\frac{1}{2\sigma^2} [(n-1)s^2 + n(\bar{y} - \mu)^2] \bigg) d\sigma^2 \ \ \ (\text{by (5.2)})
$$

이 적분은 치환적분을 통해 구할 수 있습니다. $A = (n-1)s^2 + n(\mu - \bar{y})^2$일 때, 

$$
z = \frac{A}{2\sigma^2}
$$

에 대한 적분으로 치환하면, 그 결과는 비정규화(normalizing constant가 없는 pdf를 말합니다!) 감마 pdf를 적분한 것입니다.

\begin{align}
p(\mu|y) \propto A^{-n/2} \int_0^{infty} z^{(n-2)/2}exp(-z)dz \newline
\propto [(n-1)s^2 + n(\mu - \bar{y})^2]^{-n/2} \newline
\propto \bigg[ 1 + \frac{n(\mu - \bar{y})^2}{(n-1)s^2} \bigg]^{-n/2}
\end{align}

그리고 이것은 $t_{n-1}(\bar{y}, s^2/n)$의 pdf 형태입니다.

다른 방식으로 생각해보면, 지금까지 우리는 ($\mu, log \sigma$) 상의 noninformative 균등 사전 분포 가정 하에서 $\mu$에 대한 사후분포가 다음과 같은 형식임을 알아낸것입니다.

$$
\frac{\mu - \bar{y}}{s / \sqrt{n}}\bigg| \sim t_{n-1}
$$

$t_{n-1}$은 location = 0, scale = 1이고 $n-1$의 자유도를 가지는 표준 t 분포입니다. 

이 주변 사후 분포는 표본 이론과 또다른 흥미로운 비교점을 보여줍니다. 표본 분포 $p(y|\mu, \sigma^2)$하에서, 다음과 같은 관계가 성립됩니다.


$$
\frac{\bar{y} - \mu}{s/\sqrt{n}} \bigg| \mu, \sigma^2 \sim t_{n-1}.
$$

이 [pivotal quantity](https://en.wikipedia.org/wiki/Pivotal_quantity) $(\bar{y} - \mu) / (s/\sqrt{n})$의 표분 분포는 'nuisance' 파라미터 $\sigma^2$에 의존하지 않고, 이것의 사후 분포는 데이터에 의존하지 않습니다. 일반적으로 추정값의 pivotal quantity는 데이터와 (모든 파라미터와 데이터에 독립인 표본 분포를 가지는)추정값의 복잡한 함수로 정의됩니다. 

### 미래 관찰값에 대한 사후 예측 분포

미래의 관찰값 $\tilde{y}$에 대한 사후 예측 분포는 

$$
p(\tilde{y}|y) = \int \int p(\tilde{y}|\mu, \sigma^2, y)p(\mu, \sigma^2|y)p(\mu, \sigma^2|y)d\mu d\sigma^2
$$

와 같이 나타낼 수 있습니다. 적분 안에 있는 두 가지 요소중 첫 번째는 그냥 ($\mu, \sigma^2$)가 주어졌을 때 미래 관찰값에 대한 정규분포이고, $y$에 전혀 의존하지 않습니다. 즉 사후 예측 분포에서 표본을 뽑기 위해서는, 우선 결합 사후 분포에서 $\mu, \sigma^2$을 뽑고 그것들을 활용해 $\tilde{y} \sim N(\mu, \sigma^2)$을 시뮬레이션 하면 됩니다. 

사실 위의 **$\mu$에 대한 주변 사후 분포의 해석적인 형태(Analytic form)** 파트와 같은 방식으로 사후 예측 분포를 구하면
$$
\tilde{y} \sim t_{n-1}(\bar{y}, (1+\frac{1}{n})s^2)
$$
과 같습니다. 특히 이 분포는 파라미터 $\mu, \sigma^2$를 결합 사후 분포로부터 적분을 통해 빼냄으로서 구할 수 있습니다. 우리는 이 결과를 더 쉽게 구할 수 있는데, 인수분해

$$
p(\tilde{y}|\sigma^2,y) = \int p(\tilde{y}|\mu, \sigma^2, y)p(\mu|\sigma^2,y)d\mu
$$

로 부터

$$
p(\tilde{y}|\sigma^2, y) = N(\tilde{y}|\bar{y}, (1+\frac{1}{n})\sigma^2)
$$

를 구할 수 있고, 이것은 (5.3)식($\mu|\sigma^2,y$)의 분포를 scale 팩터만 바꾼 것과 같습니다.

## 5.3 켤레 사전 분포를 사용한 Normal 데이터

### 켤레 사전 분포족(A family of conjugate prior distributions)

더 일반화된 모델을 만들기 위한 첫 번째 단계는 파라미터가 두 개인 단변량 정규 표본 모델에 noninformative 사전 분포를 고려한 켤레 사전 분포를 가정하는 것입니다. 식 (5.2)와 지금까지 배워온 것에 의하면, 켤레 사전 pdf 또한 scaled Inv-$\chi^2$인 $\sigma^2$의 주변 분포와 $\sigma^2$이 주어졌을 때의 $\mu$에 대한 조건부 분포인 정규분포의 곱의 형태 $p(\sigma^2)p(\mu|\sigma^2)$를 가져야 한다는 것을 알 수 있습니다(그래서 $\mu$의 주변 분포는 t 분포입니다). 그리고 이것은 다음과 같은 것을 활용하면 쉽게 parameterization 할 수 있습니다.

$$
\mu|\sigma^2 \sim N(\mu_0, \sigma^2/\kappa_0)
$$

$$
\sigma^2 \sim \text{Inv-}\chi^2(v_0, \sigma_0^2)
$$

즉 이 둘을 결합하면 다음과 같은 결합 사전 밀도 함수를 얻을 수 있습니다.

$$
p(\mu, \sigma^2) = p(\sigma^2)p(\mu|\sigma^2) \propto \sigma^{-1}(\sigma^2)^{-(v_0/2+1)} exp \bigg(-\frac{1}{2\sigma^2}[v_0\sigma_0^2 + \kappa_0(\mu_0 - \mu)^2] \bigg) \ \ \ \ (5.6)
$$

우리는 이것을 $N\text{-Inv-}\chi^2(\mu_0, \sigma_0^2/\kappa_0;v_0, \sigma_0^2)$의 확률 밀도 함수라고 이름붙이도록 하겠습니다. 이것의 네 가지 파라미터는 각각

1. $\mu$의 location
2. $\mu$의 scale
3. 자유도(degrees of freedom)
4. $\sigma^2$의 scale

라는 것을 확인할 수 있습니다.

$\mu|\sigma^2$의 조건부 분포에서 $\sigma^2$가 나온 것은 $\mu$와 $\sigma^2$가 그들의 결합 사전 밀도 함수에서 긴밀한 의존 관계에 다는 것을 의미합니다. 예를 들어 $\sigma^2$가 크다면, 높은 분산을 가지는 사전 분포는 $\mu$에 영향을 끼칩니다. 켤레 사전 분포가 편의를 위해 광범위하게 사용되고 있다는 점을 고려하면 이러한 의존성은 중요합니다. 그러나 다시 돌아보면, 이러한 의존성은 평균의 사전 분산이 관찰값 y의 표본 분산인 $\sigma^2$와 연관이 있다는 사실을 이해할 수 있게 됩니다. 이 방식으로 $\mu$에 대한 사전 믿음은 $y$의 측정값의 scale에 의해 조정되고, 이 scale에 대한 사전 측정값 $\kappa_0$와 같게 됩니다. 


### 결합 사후 분포, $p(\mu, \sigma^2|y)$

(5.6)에서 구한 사전 밀도 함수에 정규 likelihood를 곱하면 사후 밀도 함수를 구할 수 있습니다.

\begin{align}
p(\mu, \sigma^2|y) \propto \sigma^{-1}(\sigma^2)^{-(v_0/2+1)} exp \bigg( -\frac{1}{2\sigma^2}[v_0\sigma_0^2 + \kappa_0(\mu - \mu_0)^2] \bigg) \times \newline
\times (\sigma^2)^{-n/2}exp\bigg( -\frac{1}{2\sigma^2}[(n-1)s^2 + n(\bar{y} - \mu)^2]\bigg) \ \ \ \ \ \ \ (5.7)
\end{align}

$$
= N\text{-Inv-}\chi^2(\mu_n, \sigma_n^2/\kappa_n;v_n, \sigma_n^2)
$$

이것에 조금 조작을 가하면

\begin{align}
\mu_n = \frac{\kappa_0}{\kappa_0 + n} + \frac{n}{\kappa_0 + n} \bar{y} \newline
\kappa_n = \kappa_0 + n \newline
v_n = v_0 + n \newline
v_n \sigma_n^2 = v_0\sigma_0^2 + (n-1)s^2 + \frac{\kappa_0n}{\kappa_0 + n}(\bar{y} - \mu_0)^2
\end{align}

를 구할 수 있습니다.

   
  사후 분포의 파라미터들은 사전 정보와 데이터 안에 포함되어있는 정보를 결합한 것입니다. 예를 들어 $\mu_n$은 사전 평균과 표본 평균의 두 가지 정보의 상대적인 정확도에 의해 결정된 가중치를 가지는 가중 평균입니다. 사후 자유도인 $v_n$은 사전 자유도에 표본 크기를 더한 것과 같습니다. 사후 제곱합 $v_n\sigma_n^2$은 사전 제곱합과 표본 제곱합을 합한 값에 표본 평균과 사전 평균의 차이에 의해 발생한 불확실성을 추가적으로 더한 값과 같습니다.

### 조건부 사후 분포, $p(\mu|\sigma^2,y)$

$\mu|\sigma^2$의 조건부 사후 분포는 (5.7)과 $\sigma^2$를 상수로 하면서 비례합니다.

$$
\mu|\sigma^2, y \sim N(\mu_n, \sigma^2/\kappa_n)
$$

$$
= N\bigg(\frac{\frac{\kappa_0}{\sigma^2}\mu_0 + \frac{n}{\sigma^2}\bar{y}}{\frac{\kappa_0}{\sigma^2} + \frac{n}{\sigma^2}}, \frac{1}{\frac{\kappa_0}{\sigma^2} + \frac{n}{\sigma^2}} \bigg) \ \ \ \ \ (5.8)
$$

이 때 $\sigma$는 고정된 값으로 취급됩니다.

### 주변 사후 분포, $p(\sigma^2|y)$

$\sigma^2$의 주변 사후 밀도 함수는 식 (5.7)에 따라 scaled Inv-$\chi^2$입니다.

$$
\sigma^2|y \sim \text{Inv-}\chi^2(v_n, \sigma_n^2) \ \ \ \ \ (5.9)
$$

### 결합 사후 분포에서 표본 추출하기

결합 사후 분포에서 표본을 추출하기 위헤서는 이전 섹션에서 한 것과 같이 우선 (5.9)의 주변 사후 분포에서 $\sigma^2$를 뽑고, (5.8)의 정규 조건부 사후 분포에서 앞에서 뽑힌 $\sigma^2$를 활용해 $\mu$를 뽑으면 됩니다.

### $\mu$의 주변 사후 분포의 해석적 형태(Analytic form)

결합 사후 분포를 $\sigma^2$에 대해 적분하면, 이전 장과 정확히 유사한 방식으로 $\mu$의 주변 사후 밀도 함수를 구할 수 있습니다.

$$
p(\mu|y) \propto \bigg(1 + \frac{\kappa_n(\mu - \mu_n)^2}{v_n \sigma_n^2} \bigg)^{-(v_n+1)/2}
$$

$$
= t_{v_n}(\mu|\mu_n, \sigma_n^2/\kappa_n)
$$

### 5.4 범주형 데이터의 다항 분포

3-1장에서 다뤘던 이항 분포는 두 개 이상의 결과물이 나올 수 있는 분포로 일반화할 수 있습니다. 다항 표본 분포는 각각의 관찰값이 $k$개의 가능한 결과물 중 하나인 데이터를 표현하는데 사용됩니다. 만약 $y$가 각각의 결과물이 나오는 관찰값의 갯수라면

$$
p(y|\theta) \propto \prod_{j=1}^k \theta_j^{y_j}
$$

이고, 이 때 확률들의 합 $\sum_{j=1}^k\theta_j$는 1입니다. 그리고 이 분포는 보통 암묵적으로 관찰값인 $\sum_{j=1}^k y_j = n$에 조건부입니다. 켤레 사전 분포는 Dirichlet라고 알려진 베타 분포의 다변량 일반화 분포입니다.

$$
p(\theta|\alpha) \propto \prod_{j=1}^k \theta_j^{a_j-1}
$$

그리고 이 분포는 음수가 아닌 $\sum_{j=1}^k \theta_j = 1$인 $\theta_j$로 제한됩니다. 결과로 나오는 $\theta_j$dml 사후 분포는 파라미터가 $\alpha_j + y_j$인 Dirichlet 분포입니다.

   사전 분포는 수학적으로 j번째 범주에서 온 관찰값 $\alpha_j$의 $\sum_{j=1}^k a_j$에서 나온 결과인 likelihood와 같습니다. 이항 분포의 경우와 마찬가지로 여러가지 그럴듯한 noninformative Dirichlet 사전 분포가 있습니다. 그 결과로 나온 사후 분포는 만약 최소 하나의 관찰값이 각각의 k 범주에서 나왔다면 유효합니다. 그래서 각각의 $y$값의 원소는 양수입니다. 

많은 설문조사의 질문들의 결과를 동시에 분석하는 것과 같은 복잡한 문제에서는 다항 범주의 수와 파라미터들이 너무 크고, 모델에 추가적인 구조를 추가하지 않고는 적당한 양의 데이터에 대해서도 유용한 분석을 하기 어렵습니다. 이전에는 추가적인 정보가 사전 분포나 표본 모델을 통해 들어갔습니다. informative 사전 분포는 뒤에 배울 계층적 모델링(hierarchical modeling)을 사용함으로서 복잡한 문제에 대한 추론을 향상시켜줍니다. 또한 loglinear 모델이 여러개의 설문조사 질문을 교차 분류한 결과인 다항 파라미터들에 구조를 추가하는 것에 이용될 수 있습니다.


