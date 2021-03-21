---
title: A First Course in Bayesian Statistical Methods Chapter 4 . One parameter model, 3. Normal Model
date: 2021-03-18T17:33:01+09:00
categories: ["A First Course In Bayesian Statistical Methods"]
tags: ["Bayesian", "Python"]
---
# A First Course in Bayesian Statistical Methods

## 4. 파라미터가 하나인 정규 모델(One-parameter Normal Model)

## 4.1 데이터와 사전 정보 사이의 타협(compromise)으로서의 사후 분포

사후 확률 분포는 현재 가지고 있는 파라미터 $\theta$에 대한 모든 정보를 가지고 있습니다. 이상적으로는 전체 사후 분포 $p(\theta|y)$를 만드는 것이 가장 좋을 것입니다. 베이지안 접근 방법의 가장 큰 장점은 아무리 어떤 어려운 변환 후에도 사후 추론의 요약 통계량을 구할 수 있을 만큼 유연하다는 것입니다. 

 그러나 많은 실용적인 목적에서는 분포의 다양한 요약 통계량이 필요합니다. 자주 쓰이는 위치에 대한 요약 통계량은 평균, 중앙값, 그리고 최빈값 등이 있습니다. 분산은 표준 편차 또는 사분범위(InterQuantile Range, IQR = Q3 - Q1)를 보통 사용합니다. 각각의 요약 통계량은 분포에 대한 그들만의 해석이 존재합니다. 예를 들어 평균은 파라미터에 대한 사후 기댓값이고 최빈값은 데이터(그리고 모델)이 주어졌을 때 파라미터가 가질 수 있는 하나의 "가장 가능성 높은" 값이 됩니다. 앞으로 공부할 것에서 알 수 있겠지만, 실제적인 추론의 대부분은 정규 근사의 사용에 의존합니다. 이들은 자주 $\theta$에 대칭 변환을 적용함으로써 향상되고, 여기에서 평균과 표준편차가 중요한 역할을 합니다. 최빈값은 평균이나 중앙값보다 계산이 더 쉬운 경우가 많기 때문에, 더 복잡한 문제에 대한 computational strategies에 중요합니다. 사후 분포가 베타 분포와 같이 닫힌 형태(closed form)일 때, 사후 분포의 평균, 중앙값, 표준편차와 같은 요약통계량 또한 대부분 닫힌 형태로 만드는 것이 가능합니다.
 
 예를 들어 beta(y+1, n-y+1)의 경우는 평균이 $\frac{y+1}{n+2}$, 최빈값은 $\frac{y}{n}$ 이고 이들은 $\theta$에 대한 최대 우도(maximum likelihood)이면서 최소 분산을 가지는 불편 추정량(unbiased estimate)을 바라보는 두 가지 관점으로 잘 알려져있습니다. 

## 4.2 정규분포에서 알려진 분산을 사용해 평균 추정하기

정규분포는 대부분의 통계적 모델링의 기본입니다. 중심극한정리(Central Limit Theorem, CLT)는 많은 통계적인 문제에서 해석적으로 덜 편리한 실제 우도(likelihood)의 근사로써 정규 우도를 사용하는 것을 결정하는데 도움을 줍니다. 또한 심지어 정규분포가 그 자체로는 잘 맞지 않는 모델을 제공할 때에도 유한한 분포의 결합을 포함하는 더 복잡한 모델의 한 요소로써 유용합니다. 지금부터는 정규 모델이 적절하다는 가정을 하고 간단한 베이지안 추론을 해보겠습니다. 우선 하나의 데이터 포인트에서 결과를 도출하고, 많은 데이터 포인트의 일반적인 케이스로 확장해보도록 하겠습니다.

### 한 데이터 지점의 우도(likelihood)

가장 간단한 첫 번째 케이스는 파라미터로 평균 $\theta$와 분산 $\Sigma^2$를 가지는 정규분포의 한 스칼라 관찰값 y에 대해 생각해보는 것입니다. 표본 분포는 다음과 같습니다.

$$
p(y|\theta) = \frac{1}{\sqrt{2\pi} \sigma} e^{-\frac{1}{2\sigma^2}(y-\theta)^2}
$$

### 켤레 사전분포와 사후 분포

  이러한 우도를 $\theta$에 대한 함수로 생각해보면, exponential의 $\theta$의 이차식(quadratic) 제곱입니다. 그렇기 때문에 켤레 사전 분포들의 모양은 다음과 같을 것입니다.

$$
p(\theta) = e^{A\theta^2 + B\theta + C}
$$

이것을 파라미터의 형태로 나타내면 

$$
P(\theta) \propto exp \bigg( -\frac{1}{2\tau_0^2}(\theta - \mu_0)^2 \bigg);
$$

이고, 따라서 [하이퍼 파라미터](https://wikidocs.net/30995)가 $\mu_0$과 $\tau_0^2$인 $\theta \sim N(\mu_0, \tau_0^2)$입니다. 보통은 이러한 최초 모델의 하이퍼 파라이터는 알려진 것으로 가정합니다.

 켤레 사전 확률 밀도는 $\theta$에 대한 사후 분포가 exponential의 이차식 제곱꼴이고 따라서 정규분포라는 점을 알려줍니다. 그러나 이것의 특정한 형태를 알기 위해서는 대수적인(algebra) 스킬이 필요합니다. 사후 확률 밀도 함수에서 $\theta$를 제외한 모든 변수가 상수항으로 취급되고, 다음과 같은 조건부 확률 밀도 함수를 만들어냅니다.
 
$$
p(\theta|y) \propto exp \bigg(-\frac{1}{2} \bigg( \frac{(y-\theta)^2}{\sigma^2} + \frac{(\theta - \mu_0)^2}{\tau_0^2} \bigg) \bigg)
$$

exponential 안에 있는 것들을 펼치고 같은 변수끼리 모은 다음 $\theta$의 제곱꼴로 만들면 

$$
p(\theta|y) \propto exp \bigg(-\frac{1}{2\tau_1^2}(\theta - \mu_1)^2 \bigg) \ \ \ \ (4.1)
$$

이고,  $\theta|y \sim N(\mu_1, \tau_1^2)$ 입니다. 이 때 파라미터 $\mu_1, \tau_1$은 다음과 같습니다.

$$
\mu_1 = \frac{\frac{1}{\tau_0^2}\mu_0 + \frac{1}{\sigma^2}y}{\frac{1}{\tau_0^2} + \frac{1}{\sigma^2}} \ \ \ \text{and} \ \ \ \frac{1}{\tau_1^2} = \frac{1}{\tau_0^2} + \frac{1}{\sigma^2} \ \ \ \ (4.2)
$$

### 사전, 그리고 사후 분포의 정확도(Precision)

정규 분포를 다룰 때, 정확도(Precision)라고 불리는 분산의 역수가 중요한 역할을 합니다. 위에 있는 식을 보면, 정규 분포 데이터와 정규 사전 분포의 경우에 사후 정확도인 $\frac{1}{\tau_1^2}$은 알려진 값인 사전 정확도 $\frac{1}{\tau_0^2}$과 데이터의 정확도 $\frac{1}{\sigma^2}$의 합으로 나타납니다.

사후 평균 $\mu_1$에 대해서는 다양한 해석 방법이 있습니다. (4.2)식에서, 사후 평균은 사전 평균과 관찰된 값 $y$를 정확도에 비례한 가중 평균으로 표현됩니다. 다른 방법으로는 사전 평균을 관찰된 $y$ 값 쪽으로 조정하는 것으로 표현할 수 있습니다. 

$$
\mu_1 = \mu_0 + (y - \mu_0) \frac{\tau_0^2}{\sigma^2 + \tau_0^2}
$$

아니면 사전 분포 쪽으로 데이터가 깎여나간다고도 볼 수 있겠죠

$$
\mu_1 = y - (y - \mu_0) \frac{\sigma^2}{\sigma^2 + \tau_0^2}
$$

각각의 식은 사후 평균이 사전 평균과 관찰된 데이터 사이의 어떤 절충된 값이라는 것을 보여줍니다.

극단적인 값에서 사후 평균은 사전 평균 또는 관찰된 데이터와 일치합니다.

\begin{align}
\mu_1 = \mu_0 \ \ \ if \ y=\mu_0 \ \ or \ \ \tau_0^2 = 0; \newline
\mu_1 = y \ \ \ if \ y = \mu_0 \ \ or \ \ \sigma^2 = 0
\end{align}

  만약 $\tau_0^2 = 0$이라면, 사후 분포는 데이터보다 무한대로 더 정확합니다. 그렇기 때문에 사후 분포와 사전 분포는 동일하고 $\mu_0$ 값으로 한데 모입니다. 만약 $\sigma^2 = $이라면, 데이터는 완벽하게 정확하고 사후 분포는 관찰된 값 $y$와 동일합니다. 만약 $y = \mu_0$이라면, 사전 평균과 데이터 평균이 같다는 것을 의미하고, 사후 평균은 반드시 이 지점으로 떨어집니다.

### 사후 예측 분포

미래 관찰값 $\tilde{y}$에 대한 사후 예측 분포는 $p(\tilde{y}|y)$이고, 다음과 같이 적분을 통해 계산할 수 있습니다.

$$
p(\tilde{y}|y) = \int p(\tilde{y}, \theta|y)d\theta
$$

$$
= \int p(\tilde{y}|\theta, y) p(\theta|y)d\theta 
$$

$$
= \int \underbrace{p(\tilde{y}|\theta)}_{N(\theta,\sigma^2)} \underbrace{p(\theta|y)d\theta}_{N(\mu_1, tau_1^2)} 
$$

$$
\propto \int \ \ \ exp \bigg(-\frac{1}{2\sigma^2}(\tilde{y} - \theta)^2 \bigg) exp \bigg(- \frac{1}{2\tau_1^2}(\theta-\mu_1)^2 \bigg) d\theta
$$

첫 번째 줄은 $\theta$가 주어졌을 때 미래 관찰값 $\tilde{y}$의 분포가 과거의 데이터 $y$에 의존하지 않는다는 것(즉 조건부 독립)때문에 성립합니다. 이변량 정규 분포(bivariate normal distribution)의 특성을 이용하면 $\tilde{y}$의 분포를 더 쉽게 구할 수 있습니다. 적분 안에 있는 곱은 exponential의 $(\tilde{y}, \theta)$의 이차식 제곱 형태입니다. 따라서 $\tilde{y}, \theta$는 결합 정규 사후 분포이고 $\tilde{y}$의 주변 사후 분포(marginal posterior distribution) 또한 정규분포입니다.

이제 사후분포가 $E(\tilde{y}|\theta) = \theta, var(\tilde|\theta) = \sigma^2$라는 것을 활용해 사후 예측 분포의 평균과 분산을 구해보도록 하겠습니다.

수리통계학에서 배운 내용을 활용해 평균은 다음과 같이 구할 수 있고,

$$
E[\tilde{y}|y] = E[\underbrace{E[\tilde{y}|y]}_{ = \theta}|y] = E[\theta|y] = \mu_1
$$

분산은 다음과 같습니다.

$$
var[\tilde{y} | y ]
$$

$$
=E[\underbrace{var[\tilde{y}|\theta,y]}_{=\sigma^2}|y] + var[\underbrace{E[\tilde{y}|\theta,y]}_{=\theta}|y]
$$

$$
= E[\sigma^2|y] + var[\theta|y]
$$

$$
= \sigma^2 + \tau_1^2
$$



따라서 $\tilde{y}$의 사후 예측 분포의 평균은 $\theta$의 사후 평균 $\mu_1$과 같고, 사후 분산은 모델의 분산 $\sigma^2$와 $\theta$의 사후 불확실성인 $\tau_1^2$을 더한 것과 같습니다.

### 여러개의 관찰값을 가지고 있는 정규분포 모델

하나의 관찰값을 가지고 있을 때의 정규 모델인 위와 같은 경우는 i.i.d. 표본 관찰값 $y = (y_1, ..., y_n)$들을 가지고 있는 현실적인 상황으로 쉽게 확장될 수 있습니다.

앞에서 만들었던 사후 예측 분포의 식을 다음과 같이 확장해봅시다.

$$
p(\theta|y) \propto p(\theta) p(y|\theta)
$$

$$
= p(\theta) \prod_{i=1}^n p(y_i|\theta) 
$$

$$
\propto exp \bigg( -\frac{1}{2\tau_0^2} (\theta - \mu_0)^2 \bigg) \prod_{i=1}^n exp \bigg( -\frac{1}{2\sigma^2}(y_i - \theta)^2 \bigg)
$$

$$
\propto exp \bigg( -\frac{1}{2}\bigg(\frac{1}{\tau_0^2}(\theta - \mu_0)^2 + \frac{1}{\sigma^2} \sum_{i=1}^n (y_i - \theta)^2 \bigg) \bigg).
$$

그리고 이 식을 단일 관찰값일 때와 비슷한 방식으로 대수적으로 단순화하면

$$
p(\theta|y_1, ..., y_n) = p(\theta|\bar{y}) = N(\theta|\mu_n, \tau_n^2), 
$$

이고 이 때 $\mu_n, \tau_n^2$은

$$
\mu_n = \frac{\frac{1}{\tau_0^2}\mu_0 + \frac{n}{\sigma^2} \bar{y}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}} \ \ \ \ \ \ \text{이고,} \frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}
$$

입니다.

여기서 볼 수 있는 것은 사후 분포가 오직 표본 평균인 $\bar{y} = \frac{1}{n}\Sigma_i y_i$를 통해서만 $y$에 의존한다는 것을 보여줍니다. 즉 $\bar{y}$는 이 모델에서 충분 통계량(sufficient statistic)입니다. 사실, $\bar{y}|\theta \sim N(\theta, \sigma^2/n)$이기 때문에, 이 결과를 바로 단일 관찰값인 경우로 적용할 수 있습니다.($\bar{y}$를 단일 관찰값 취급하면 됩니다.)

또한, 데이터 $y_1, y_2, ... y_n$을 하나하나 추가하고 각각의 단계에서 구해진 사후 분포를 다음 단계의 사전 분포로 사용하면 위와 같은 결과룰 얻을 수 있습니다.  

사후 평균과 분산의 표현에서, 사전 정확도인 $1/\tau_0^2$와 데이터 정확도 $n\sigma^2$는 같은 역할을 합니다. 따라서 만일 $n$이 크다면, 사후 분포에서 $\sigma^2$과 표본 평균 $\bar{y}$의 영향력이 더 크게 됩니다. 예를 들어 만약 $\tau_0^2 = \sigma^2$라면, 사전 분포는 값이 $\mu_0$인 하나의 추가적인 관찰값으로써 다른 관찰값들과 같은 가중치를 가지게 됩니다. 예를 들어 n이 고정된 값일 때 $\tau_0 \rightarrow \infty$ 또는 $\tau_0^2$이 고정됐을 때 $n \rightarrow \infty$와 같은 특별한 경우는 다음과 같게 됩니다.

$$
p(\theta|y) \approx N(\theta|\bar{y}, \sigma^2/n)
$$

이것은 실전에서 사전 믿음이 $\theta$의 범위에서 넓게 퍼져있고, 데이터의 우도(likelihood)가 중요할 때면 언제나 좋은 근사가 됩니다.

## 4.3 평균을 알고, 분산을 모를 때의 정규 분포

평균을 알고 분산을 모를 때의 정규 모델은 그 자체로 중요하기 보단, 더 어려운 모델, 특히 바로 다음 장에 배울 평균과 분산을 모두 모를 때의 정규 분포를 만들 때 필요한 한 파트로서 중요합니다. 추가적으로, 평균을 알고 분산을 모를 때의 정규분포는 스케일 파라미터를 추정하는 입문적인 예시를 제공합니다.

**구하고자 하는 사후 분포** 
$p(y|\theta, \sigma^2) = N(y|\theta, \sigma^2)$,        ($\theta$ : known, $\sigma^2$ : unknown)

**$\theta$는 상수이고, $\sigma^2$는 변수일 때 데이터의 우도(likelihood)**

$$
p(y|\sigma^2) \propto \sigma^{-n} exp \bigg( -\frac{1}{2\sigma^2} \sum_{i=1}^n(y_i - \theta)^2 \bigg)
$$

$$
= (\sigma^2)^{-n/2} exp \bigg(-\frac{n}{2\sigma^2} v \bigg)
$$


이 때 $v$는 충분 통계량인

$$
v = \frac{1}{n} \sum_{i=1}^{n}(y_i - \theta)^2
$$

입니다.

**이에 대응하는 켤레 사전 분포**

하이퍼 파라미터 $(\alpha, \beta)$를 가지는 inverse-gamma 분포인
$$
p(\sigma^2) \propto (\sigma^2)^{-(\alpha + 1)} e^{-\beta / \sigma^2}
$$

이것을 간단하게 [파라미터화](https://en.wikipedia.org/wiki/Parametrization_(geometry)) 하면 scale이 $\sigma_0^2$이고 자유도가 $v_0$인 scaled inverse-$\chi^2$ 분포입니다. 즉 $\sigma^2$의 사전 분포는 $X$가 $\chi_{v_0}^2$ 분포를 따르는 확률변수인 $\sigma_0^2 v_0/X$의 분포를 가집니다. 그리고 이것을 다음과 같이 쓰도록 하겠습니다(표준적인 노테이션은 아닙니다!)

$$
\sigma^2 \sim \text{Inv-}\chi^2(v_0, \sigma_0^2)
$$

그리고 이것의 pdf는 

$$
p(\sigma^2) = \frac{2^{-v_0/2}}{\Gamma(v_0/2)} (\sigma_0^2)^{v_0} (\sigma^2)^{-(v_0/2+1)} e^{-v_0 \sigma_0^2/(2\sigma^2)}
$$

이고, 사전분포와 데이터의 우도를 결합한 사후 분포를 구하면,

$$
p(\sigma^2|y) \propto p(\sigma^2)p(y|\sigma^2)
$$

$$
\propto \underbrace{\bigg(\frac{\sigma_0^2}{\sigma^2}\bigg)^{v_0/2+1} exp\bigg(-\frac{v_0\sigma_0^2}{2\sigma^2} \bigg)}_{prior} \underbrace{(\sigma^2)^{-n/2} exp \bigg(-\frac{n}{2} \frac{v}{\sigma^2} \bigg)}_{\text{likelihood of data}} 
$$

$$
\propto (\sigma^2)^{-((n+v_0)/2+1)} exp \bigg(-\frac{1}{2\sigma^2}(v_0 \sigma_0^2 + nv) \bigg)
$$

입니다. 이것은 위에서 본 $\text{Inv-}\chi^2$분포의 pdf와 같은 형태인 것을 알 수 있습니다. 즉

$$
\sigma^2|y \sim \text{Inv-}\chi^2 \bigg(v_0 + n, \frac{v_0 \sigma_0^2 + nv}{v_0 + n} \bigg)
$$

이 됩니다.

이것의 파라미터를 해석해보면, scale 파라미터인 $\frac{v_0 \sigma_0^2 + nv}{v_0 + n}$은 사전 분포의 scale과 데이터의 scale을 각각의 자유도($v_0, n$)를 가중치로 한 가중평균과 같고, 자유도 파라미터 $v_0 + n$은 사전 분포와 데이터의 자유도를 합한 값과 같습니다. 즉 사전 분포는 평균 제곱 편차가 $\sigma_0^2$인 $v_0$개의 관찰값이 주는 것과 같은 정보를 제공한다고 생각할 수 있습니다.

