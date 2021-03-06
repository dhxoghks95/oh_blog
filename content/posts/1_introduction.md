---
title: A First Cource in Bayesian Statistical Methods Chapter 1. Introduction
author : 오태환
date: 2021-02-09T19:59:51+09:00
categories : ["A First Course In Bayesian Statistical Methods"]
tags : ["Bayesian", "Python"]

---

# A First Course in Bayesian Statistical Methods

## 1. 왜 베이지안을 배워야 하는가?

### 1.1 Introduction

우리는 알려지지 않은 수량(quantity)에 대한 우리의 정보와 믿음을 표현하기 위해 비공식적으로 확률을 자주 사용합니다. 하지만 정보를 표현하기 위해 확률을 사용하는 것은 공식적인 것으로 만들 수 있죠. 정확한 수학적인 관점에서, 확률은 확률과 정보 사이에 관계가 있다는 합리적인 믿음을 수식적으로 표현할 수 있습니다. 그리고 베이즈 법칙은 새로운 정보를 투입해 믿음을 업데이트하는 합리적인 방법론을 제공합니다. 베이즈 법칙을 통한 귀납적인(개별적인 특수한 사실이나 현상에서 그러한 사례들이 포함되는 일반적인 결론을 이끌어내는 추론) 학습 과정을 바로 **베이지안 추론**이라고 부릅니다.

 더 일반적으로 표현하자면, **베이지안 방법론**들은 **베이지안 추론**의 이론을 사용한 데이터 분석 툴입니다. 이전에 말했던 귀납적인 의미로써의 해석에 더해 베이지안 방법론들은 다음과 같은 것들을 제공합니다.

* 훌륭한 통계적인 특성들을 활용한 모수 추정
* 관찰된 데이터에 대한 간결한 설명
* Missing Data와 미래 데이터에 대한 예측
* 모델의 추정, 선택, 그리고 검증을 위한 compuational framework

즉 베이지안 방법론의 활용 범위는 그 기반인 귀납적인 영역을 뛰어넘습니다. 우리는 이 책을 통해 다양한 추론적인, 그리고 통계적인 업무를 위한 베이지안 방법론의 광범위한 활용들을 알아볼 것입니다. 우리는 이 챕터를 베이지안 학습의 기본적인 재료들을 베이지안 방법론이 실생활에 사용되는 각기 다른 방법들을 예시로 들며 소개하면서 시작해보도록 하겠습니다.

### *베이지안 학습*

통계적인 귀납법은 모집단의 일반적인 특성들을 모집단으로부터 뽑은 샘플을 통해 학습하는 과정입니다. 모집단 특성에 대한 수적인(numerical) 값은 보통 파라미터 $\theta$로 표현되고, 샘플에 대한 설명은 데이터셋 $\mathcal{y}$로 만들어집니다. 데이터셋이 얻어지기 전에는 모집단의 특성과 데이터셋 둘 모두의 수적인 값들은 불확실합니다. 데이터셋 $y$가 얻어진 후에는, 그것으로 부터 얻어진 정보가 모집단의 특성에 대한 우리의 불확실성을 줄이는데 사용될 수 있습니다. 이러한 불확실성의 변화를 정량적으로 측정하는 것(Quantifying)이 베이지안 추론의 목적입니다.

 표본 공간 $\mathcal{Y}$는 모든 가능한 데이터셋들의 집합이고, $\mathcal{y}$는 그 하나하나의 데이터셋입니다. 파라미터 공간 $\Theta$는 모든 가능한 파라미터 값들의 집합이고, 우리는 여기에서 실제 모집단의 특성을 가장 잘 나타내는 값을 찾아내길 바랍니다. 베이지안 학습의 구현은 $\mathcal{y}$와 $\theta$의 joint beliefs를 $\mathcal{Y}, \Theta$상에서의 확률 분포의 형태로 표현되는 수식으로 나타내는 것 부터 시작됩니다. 

1. 각각의 수적인 값 $\theta \in \Theta$에 대해, 우리의 사전 분포 $p(\theta)$는 $\theta$가 실제 모집단의 특성을 표현한다는 우리들의 믿음을 나타냅니다.

2. 각각의 $\theta \in \Theta$과 $\mathcal{y} \in \mathcal{Y}$에 대해, 우리의 샘플링 모델 $p(y | \theta)$는 만약 우리가 $\theta$가 참값이란 것을 안다면, $y$가 우리의 연구의 결과라는 우리의 믿음을 나타냅니다. 

데이터 $y$가 주어졌을 때, 마지막 단계는 $\theta$에 대한 우리들의 믿음들을 업데이트하는 것입니다.

3. 각각의 수적인 값 $\theta \in \Theta$에 대해, 우리의 사후 분포 $p(\theta | y)$는 관찰된 데이터셋 $y$가 주어졌을 때, $\theta$가 참값이라는 우리의 믿음을 나타냅니다.

사후 분포는 사전 분포와 샘플링 모델으로 부터 베이즈 법칙을 통해 얻어집니다.

$$
p(\theta | y) = \frac{p(y|\theta)p(\theta)}{\int_{\theta}p(y|\tilde{\theta})p(\tilde{\theta})}
$$

베이즈 법칙은 우리의 믿음이 어떻게 되야하는지를 말해주는 것이 아니라, 새로운 정보를 관찰했을 때 우리의 믿음이 어떻게 변해야 하는지를 말한다는 것을 기억하는 것이 중요합니다.

### 1.2 왜 베이즈인가?

Cox(1946, 1961)와 Savage(1954, 1972)의 수학 논문들은 다음을 증명했습니다.

> 만약 $p(\theta)$와 $p(y | \theta)$가 합리적인 사람의 믿음을 표현한다면, 베이즈 법칙은 새로운 정보 $y$가 주어졌을 때 $\theta$에 대한 이 사람의 믿음을 업데이트하는 최적의 방법이다.

이러한 결론은 정량적인 학습 방법으로써의 베이지안 법칙의 활용에 강력한 이론적 정당성을 부여했습니다. 그러나 실제 데이터 분석 상황에서 무엇이 우리의 사전 믿음인지를 정확한 수식으로 만드는 것은 어려운 일일 수 있습니다. 그래서 $p(\theta)$는 약간의 임시적인 방편이나 계산이 편리한 것들로 자주 골라졌습니다. 그렇다면 베이지안 데이터 분석의 정당성은 어떤 것일까요?

 샘플링 모델에 대한 유명한 인용구를 들어보겠습니다.
 > 모든 모델은 틀렸다, 그러나 몇몇은 유용하다(Box and Draper, 1987, pg. 424)
 
 비슷하게 $p(\theta)$는 만일 우리의 믿음을 정확하게 표현하지 못한다면 틀린 것으로 보일 수 있습니다. 그러나 이것은 $p(\theta | y)$가 쓸모없다는 것을 의미하지는 않습니다. 만일 $p(\theta)$가 우리의 믿음에 근사한다면, $p(\theta | y)$가 $p(\theta)$ 하에서 최적이라는 사실은 $p(\theta | y)$ 또한 일반적으로 우리의 사후 믿음이 어떨지에 대한 훌륭한 근사를 제공한다는것을 의미합니다. 다른 상황에서는 우리가 관심있는 것이 우리의 믿음이 아닐 수도 있습니다. 그 대신 베이즈 법칙을 데이터가 다양한 사람들의 믿음이 사전 의견들과 어떻게 달라지는지 업데이트하는데 사용하는데 관심이 있을 수 있습니다. 우리가 특별히 관심 가지는 것은 약한 사전 정보(Weak Prior Information)를 사용한 사후 믿음입니다. 이것은 파라미터 공간 상에서의 큰 범위에 거의 동일하게 확률을 할당하는 "퍼진" 사전 분포의 사용에서 기인했습니다. 
 마지막으로, 많은 복잡한 통계적인 문제에서 추정이나 추론을 위한 명확한 non-베이지안 방법론이 딱히 없습니다. 이러한 상황에서 베이즈 법칙은 추정 과정으로 사용될 수 있습니다. 그리고 이 과정의 퍼포먼스는 non-베이지안 방법을 사용해 평가될 수 있습니다. 많은 상황에서(심지어 non-베이지안 목적에서도) 베이지안 또는 근사적인 베이지안 방법들은 아주 잘 작동한다는 것이 밝혀졌습니다.
 다음에 소개할 두 가지 예시는 약한 사전 믿음(우리나 다른 사람의 사전 믿음을 러프하게 표현하는 사전 분포)을 사용한 베이지안 추론이 통계적 추론에 어떻게 광범위하게 사용될 수 있는지를 보여주기 위함입니다.

### 1.2.1 희박한 사건의 확률 추정하기

우리가 작은 도시에서의 전염병 출현 확률에 관심있다고 가정합시다. 출현 가능성이 높을 수록, 공공 의료계에서 더욱 주의를 요할 것입니다. 20명의 작은 랜덤 샘플이 감염되었는지 검사받을 것입니다.

### *파라미터와 표본 공간*

관심있는 것은 도시 안에서 감염된 사람들의 비율인 $\theta$입니다. 대충 말하자면, 파라미터 공간은 0과 1 사이에 있는 모든 숫자들입니다. 데이터 $y$는 샘플 안에서 감염된 모든 사람의 수를 기록합니다. 파라미터 공간과 표본 공간은 그렇다면 다음과 같습니다.

$$
\Theta = [0,1] \ \ \ \ \ \ \mathcal{Y} = \{0,1,2,...,20\}.
$$

### *샘플링 모델*

샘플을 얻기 전에는 샘플 안에 있는 감염된 사람들의 숫자는 미지수입니다. 우리는 변수 $Y$로 이러한 추후에 결정될 값을 표시하겠습니다. 만약 $\theta$의 값이 알려진 값이라면, $Y$에 대한 합리적인 샘플링 모델은 $\text{binomial}(20,\theta)$라는 확률 분포가 될 것입니다.

$$
Y | \theta \sim \text{binomial}(20, \theta).
$$

그림 1.1의 첫 번째 패널은 $\theta$가 각각 0.05, 0.10, 0.20일 때 $\text{binomial}(20, \theta)$의 그래프입니다. 만약 예를들어 살제 감염율이 0.05라면, 샘플에 감염된 사람이 한 사람도 없을 확률(즉, $Y = 0$일 확률)은 36%입니다. 만일 실제 감염율이 0.10이거나 0.20이라면, $Y = 0$일 확률은 각각 12%, 1%입니다.


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,6))
import numpy as np
from scipy.stats import binom, beta


# Panel 1
ax1 = plt.subplot(121)


xrange = np.arange(0,20)

theta1 = [binom.pmf(x, 20, 0.05) for x in xrange]
theta2 = [binom.pmf(x, 20, 0.10) for x in xrange]
theta3 = [binom.pmf(x, 20, 0.20) for x in xrange]

ax1.bar(xrange-0.3, theta1, label = r'$\theta = 0.05$', width = 0.3)
ax1.bar(xrange, theta2, label = r'$\theta = 0.10$', width = 0.3)
ax1.bar(xrange+0.3, theta3, label = r'$\theta = 0.20$', width = 0.3)
plt.xticks([0,5,10,15,20])
plt.xlabel('number infected in the sample')
plt.ylabel('probability')
ax1.legend(loc = 'upper right')

# Panel 2

ax2 = plt.subplot(122)

xrange = np.linspace(0.0, 0.99, 99)
prior = beta.pdf(xrange,2,20)
plt.plot(xrange,prior,color = 'grey', label = r'$p(\theta)$')
posterior = beta.pdf(xrange, 2, 40)
plt.plot(xrange, posterior, color = 'k', label = r'$p(\theta | y)$')
plt.yticks([0,5,10,15])
plt.xlabel('percentage infected in the population')
plt.legend()

plt.show();
```


    
![output_26_0](https://user-images.githubusercontent.com/57588650/107350101-5c76c800-6b0c-11eb-8712-ef026d0871cd.png)
    


**그림 1.1.** 이 그래프들은 감염율 예시의 샘플링 모델, 사전분포, 사후분포입니다. 왼쪽의 그래프는 세 가지 $\theta$ 값에 따른 $\text{binomial}(20, \theta)$를 보여줍니다. 오른쪽의 그래프는 $\theta$의 사전(회색), 사후(검은색) 확률 밀도 함수를 나타냅니다.

### *사전 분포*

다른 여러 나라들의 연구에 따르면, 비슷한 도시에서의 감염률은 0.05와 0.20 사이에서 평균 0.10의 값을 보인다고 알려져있습니다. 이 사전정보는 우리의 사전 분포 $p(\theta)$로 (0.05, 0.20) 범위 안에서 기댓값이 0.10에 가까운 확률을 할당하는 분포를 사용해야 한다고 제안합니다. 그러나 이 조건을 만족하는 분포는 무수히 많고 이 중 어떤걸 사용해야하는지 우리의 제한된 사전 정보로는 구별해낼 수 없습니다. 그렇기 때문에 우리는 위에서 설명한 특성을 만족하고 computational 측면에서 편리한 $p(\theta)$를 사용할 것입니다. 구체적으로 우리는 사전 분포를 베타 분포족(family of beta distribution)에 피팅시킬 것입니다. 베타 분포는 두 개의 파라미터 a, b를 가집니다. 만일 $\theta$가 $\text{beta}(a,b)$라면, $\theta$의 기댓값은 $a/(a+b)$이고 최빈값은 $(a-1)/(a-1+b-1)$입니다. $\theta$가 감염률인 우리의 문제에서, $\theta$에 대한 사전 분포를 $\text{beta}(2,20)$의 확률 분포로 나타낼 것입니다. 이것을 기호로 나타내면 다음과 같습니다.

$$
\theta \sim \text{beta}(2,20)
$$

이 분포는 **그림 1.1**의 두 번째 패널의 회색 선으로 표현됩니다. 이 사전분포에서의 $\theta$의 기댓값은 0.09입니다. 사전 분포의 최고점은 $\theta = 0.05$일 때 입니다. 그리고 약 2/3의 영역이 0.05와 0.20 사이의 곡선 아래에서 발생합니다. 감염율이 0.10보다 낮을 사전 확률은 64%입니다.

$$
\text{E}[\theta] = 0.09 
$$

$$
\text{mode}[\theta] = 0.05 
$$

$$
\text{Pr}(\theta < 0.10) = 0.64 
$$

$$
\text{Pr}(0.05 < \theta < 0.20) = 0.66
$$

### *사후 분포*

 나중에 챕터 3에서 보겠지만, 만약 $Y|\theta \sim \text{binomial}(n, \theta)$이고, $\theta \sim \text{beta}(a,b)$이며 $Y$의 numeric 값인 $y$를 관측했을 때, 사후 분포는 $\text{beta}(a + y , b + n - y)$입니다. 우리의 연구에서 $Y$의 값이 0으로 관측했다고 가정해봅시다(샘플로 뽑힌 사람들 중 아무도 감염되지 않은 것이죠). 그렇다면 이 때 $\theta$의 사후 분포는 $\text{beta}(2,40)$입니다.
 
$$
\theta|\{\text{Y} = 0\} \sim \text{beta}(2,40)
$$

이 분포의 확률 밀도 함수는 **그림 1.1**의 두 번째 패널에서 까만 선으로 표현됩니다. 이 확률 밀도 함수는 사전 분포보다 더 왼쪽으로 이동했습니다. 그리고 더 뾰족해졌습니다. $p(\theta)$의 왼쪽으로 이동한 이유는 관찰값인 $\text{Y} = 0$이 $\theta$값이 낮을 것이라는 증거를 제공했기 때문입니다. 그리고 $p(\theta)$보다 더 뾰족해진 이유는 데이터로부터의 정보와 사전 분포가 결합되어 $p(\theta)$ 홀로 있을 때 보다 더 많은 정보를 포함하고 있기 때문입니다. 사후 분포의 첨도는 0.025이고 기댓값은 0.048입니다. $\theta < 0.10$일 사후 확률은 93%입니다.

$$
\text{E}[\theta|\text{Y} = 0] = 0.048 
$$

$$
\text{mode}[\theta | \text{Y} = 0] = 0.025 
$$

$$
\text{Pr}(\theta < 0.10 | \text{Y} = 0) = 0.93
$$

사후 분포 $p(\theta | \text{Y} = 0)$은 우리에게 전체 도시의 감염률 $\theta$의 학습을 위한 모델을 제공합니다. 이론적인 관점에서, $\theta$가 $\text{beta}(2,20)$의 분포로 표현된다는 합리적인 개인의 사전 믿음은, 이제 $\text{beta}(2,40)$의 분포로 표현된다는 믿음으로 바뀌게 됩니다. 실용적인 측면에서, 만일 우리가 $\text{beta}(2,20)$ 분포를 합리적인 사전 분포의 척도라고 받아들인다면, 우리는 $\text{beta}(2,40)$을 사후 분포의 합리적인 척도로 받아들이게 될 것입니다.

### *민감도 분석*

우리가 시 보건 당국과 조사 결과에 대해 논의한다고 가정해봅시다. 다양한 그룹의 사람들 사이에서 우리의 연구가 무엇을 함축하고있는지에 대한 논의는 다양한 사전 분포에 따른 사후 믿음들을 설명하는데 도움이 됩니다. 우리가 (2,20) 말고 다른 $(a,b)$ 값을 사용해 믿음을 $\text{beta}(a,b)$으로 표현하는 것을 고려한다고 가정해봅시다. 위에서 언급했던 것 처럼, 만약 $\theta \sim \text{beta}(a,b)$이고, $Y = y$가 주어졌을 때, $\theta$의 사후 분포는 $\text{beta}(a + y, b + n - y)$ 입니다. 사후 기댓값은 다음과 같습니다.

$$
\text{E}[\theta | Y = y] = \frac{a + y}{a + b + n} 
$$

$$
= \frac{n}{a + b + n} \frac{y}{n} + \frac{a+b}{a+b+n} \frac{a}{a+b} 
$$

$$
= \frac{n}{w + n} \bar{y} + \frac{w}{w + n} \theta_{0}
$$

이 때, $\theta_{0} = a / (a + b)$는 $\theta$의 사전 기댓값이고, $w = a + b$ 입니다. 이 식에서 우리는 사후 기댓값이 표본 평균 $\bar{y}$와 사전 기댓값 $\theta_{0}$의 가중평균이라는 것을 알 수 있습니다. $\theta$를 추정함에 있어서, $\theta_{0}$은 실제 $\theta$의 값에 대한 사전 추측을 나타냅니다. 그리고 $w$는 이 추측에 대한 우리의 확신 정도를 의미합니다. 그리고 이들을 표본의 크기에 따라 같은 스케일로 표현한 것이죠. 

![SmartSelectImage_2021-01-18-22-06-22](https://user-images.githubusercontent.com/57588650/104919337-70318180-59d9-11eb-83ff-10fc694ac126.png)

**그림 1.2.** 각기 다른 베타 사전 분포 하에서의 사후 분포들입니다. 왼쪽과 오른쪽 패널은 각각 사전 예측값의 범위와 자신있는 정도에 따른 $\text{E}[\theta | Y = 0]$와 $\text{Pr}(\theta < 0.10 | \text{Y} = 0)$의 등고선을 보여줍니다.

만일 누군가가 우리에게 사전 추측인 $\theta_{0}$와 자신감의 정도인 $w$를 제공한다면, 우리는 그들의 $\theta$에 대한 사전 믿음이 파라미터가 $a = w\theta_{0}, b = w(1-\theta_{0})$인 베타 분포를 따른다고 근사할 수 있습니다. 그들의 근사 사후 분포는 그렇다면 $\text{beta}(w\theta_{0} + y , w(1-\theta_{0}) + n + y)$로 나타나게 됩니다. 우리는 넓은 범위의 $\theta_{0}$와  $w$ 값들을 가지는 사후 분포를 계산하여 얼마나 사후 분포가 사전 의견 차이에 영향을 받는지를 찾는 *민감도 분석* 을 수행할 수 있습니다. **그림 1.2.** 는 $\theta_{0}$와 $w$의 사후 분포에 대한 영향을 두 등고선 그래프를 통해 나타냅니다. 첫 번째 그래프는 사후 기댓값 $\text{E}[\theta | Y = 0]$의 등고선 그래프이고, 두 번째 그래프는 사후 확률 $\text{Pr}(\theta < 0.10 | Y = 0)$의 등고선 그래프입니다. 두 번째 그래프는 예를 들어 만약 시 당국이 현재 감염율이 0.10보다 낮을지 확실하지 않을 때, 공중 보건에 백신을 추천해야 하는지 결정해야 할 때 사용할 수 있습니다. 예를 들어 그래프는 약한 사전 믿음(낮은 $w$값)을 가지고 있거나 낮은 사전 기댓값을 가지고 있는 사람들은 일반적으로 감염율이 0.10보다 낮을 것이란 것에 90% 이상 확신한다는 것을 보여줍니다. 반면 높은 자신감의 정도는(97.5%라고 합시다) 오직 이미 다른 도시들의 평균 감염율에 비해 감염율이 낮다고 생각하는 사람들에게만 얻을 수 있습니다. 

### *베이지안이 아닌 방법론과의 비교*

모 비율 $\theta$를 추정하는 일반적인 방법은 샘플 안에서의 감염자 비율을 나타내는 표본 평균 $\bar{y} = y / n$입니다. $y = 0$인 우리의 예시에서 이 방식으론 추정값이 0이 나오고, 도시에서 아무도 감염되지 않았다고 추정할 것입니다. 만일 우리가 이 추정값을 의사나 보건 당국에게 전달해야 한다면, 우리는 이 추정값이 표본 추출의 불확실성을 가지고 있다는 것에 대한 경고를 포함하고 싶어할 것입니다. 추정값에서 표본 추출의 불확실성을 표현하는 하나의 방식은 신뢰구간을 사용하는 것입니다. 모 비율 $\theta$에 대해 보편적으로 사용되는 95% 신뢰 구간은 *Wald interval*이라고 부르며 다음과 같은 식으로 표현됩니다.

$$
\bar{y} \pm 1.96 \sqrt{\bar{y}(1 - \bar{y}) / n}
$$

이 구간은 *Correct asymptotic frequentist coverage*를 가집니다. *Correct asymptotic frequentist coverage*란 만약 $n$이 크다면 점근적으로 95%와 같아지는 확률로 $Y$는 위의 구간 상에 $\theta$를 포함하는 $y$의 값을 가진다는 것을 의미합니다. 불행하게도 이 방법은 작은 $n$에서는 성립되지 않습니다. $n$이 20개 근처라면 구간 안에 $\theta$가 포함될 확률이 80%정도 밖에 되지 않습니다.(Agresti and Coull, 1998). 또한 $\bar{y} = 0$인 우리의 샘플에서 Wald 신뢰 구간은 구간이 아니라 한 점인 0으로 나올 것입니다. 95% 뿐만 아니라 99% 신뢰 구간에서도 Wald interval은 0이라는 한 점으로 나오겠죠. 확실하게 우리는 조사의 결과로 99.9%의 확률로 도시에 아무런 감염자가 없다는 결론을 내리고 싶진 않을 것입니다. 
 사람들은 이러한 종류의 상황을 피하기 위해 Wald interval의 다양한 대체재들을 제안해왔습니다. 베이지안이 아닌 방법 중에 잘 작동하는 신뢰 구간 중 하나는 Agresti and Coull(1998)에 의해 제안된 "조정된(adjusted)" Wald interval입니다. 식은 다음과 같습니다.
 
$$
\hat{\theta} \pm 1.96\sqrt{\hat{\theta}(1-\hat{\theta})/n}, 
$$

$$
이 때 \ \hat{\theta} = \frac{n}{n+4}\bar{y} + \frac{4}{n+4} \frac{1}{2}
$$

그렇게 의도되진 않았지만, 이 구간은 명백하게 베이지안 추론과 관련되어있습니다. 여기서 $\hat{\theta}$의 값은 $\theta = 1/2$를 중심으로 하는 약한 사전 정보를 나타내는 $\text{beta}(2,2)$ 사전 분포 하에서 $\theta$의 사후 기댓값과 동일합니다. 

모집단에서 $n$개의 랜덤 샘플이 주어졌을 때, 정석적인 모평균 $\theta$에 대한 추정량은 표본 평균인 $\bar{y}$입니다. 비록 $\bar{y}$가 일반적으로 큰 표본 크기에서는 신뢰할 수 있는 추정량이지만, 예시에서 본 것 처럼 작은 $n$에 대해서는 통계적으로 신뢰할 수 없습니다. 이 상황에서는 이것이 정확한 $\theta$의 추정량을 제공한다기 보다는 표본 데이터의 요약 통계량을 제공하는 것에 가깝죠. 
 만약 우리가 샘플 데이터의 요약 통계랑보다 $\theta$의 추정량을 얻는 것에 관심이 있다면, 다음과 같은 식으로 구하는 추정량을 고려할 것입니다.
$$
\hat{\theta} = \frac{n}{n+w}\bar{y} + \frac{w}{n+w}\theta_{0}
$$
여기서 $\theta_{0}$은 실제 $\theta$값에 대한 "최선의 추측"을 나타내고, $w$이 추측에 대해 확신하는 정도를 나타냅니다. 만약 샘플의 크기가 크다면 $\bar{y}$는 신뢰할만한 $\theta$의 추정량이 되겠죠. 추정량 $\hat{\theta}$는 $\bar{y}, \theta_{0}$에 대한 가중치가 $n$이 커짐에 따라 각각 1, 0으로 수렴한다는 점에서 장점을 가지고 있습니다. 결론적으로 $\bar{y}$와 $\hat{\theta}$의 통계적인 특성은 큰 $n$ 하에서 근본적으로 같습니다. 그러나 작은 $n$에 대해서는 $\bar{y}$의 변동성이 $\theta_{0}$에 대한 불확실성보다 더 클 것입니다. 이 상황에서 $\hat{\theta}$를 쓴다면 데이터와 사전 정보를 결합해 $\theta$에 대한 추정을 안정화시켜줄 것입니다.


 이러한 크고 작은 $n$ 모두에 대한 $\hat{\theta}$의 특성은 이 추정량이 넓은 범위의 $n$에서 유용한 $\theta$에 대한 추정량이란 것을 보여줍니다. 우리는 섹션 5.4에서 몇몇 상황 하에서 $\hat{\theta}$가 $\theta$의 추정량으로써 모든 $n$값에서 $\bar{y}$의 퍼포먼스를 능가한다는 점을 보여주면서 이것을 확인시켜줄 것입니다. 감염률 예시에서 그리고 다음 챕터에서 다시 볼 것 처럼, $\hat{\theta}$는 특정한 클래스의 사전 분포를 사용한 베이지안 추정량으로 해석될 수 있습니다. 심지어 특정한 사전 분포 $p(\theta)$가 정확하게 우리의 사전 정보를 반영하지 않는다고 해도, 그에 따른 사후 분포 $p(\theta|y)$는 여전히 안정적인 추론과 작은 샘플을 가지는 상황에서의 추정량을 제공한다는 점에서 유용합니다.

### 1.2.2 예측 모델 만들기

챕터 9에서 우리는 당뇨병 진행에 대한 예측 모델을 나이, 성별, bmi 등 기본적인 설명 변수를 사용한 함수로 만드는 작업을 수행하는 한 예시에 대해 다룰 것입니다. 여기서는 그 예시에 대한 간단한 줄거리를 설명하도록 하겠습니다. 우리는 342명의 환자들의 수치들을 가지고 있는 훈련용 데이터셋을 사용해 회귀 모델에서의 파라미터들을 추정할 것입니다. 그 다음은 추정된 회귀 모들의 예측 성능을 별도로 100명의 환자들의 "테스트" 데이터셋을 활용해 평가할 것입니다.

### *샘플링 모델과 파라미터 공간*

$Y_i$를 실험 대상 $i$의 당뇨병 진행, $x_i = (x_{i,1}, ... , x_{i,64})$를 설명변수라고 합시다. 우리는 다음과 같은 식을 통해 선형 회귀 모델을 고려해보겠습니다.
$$
Y_i = \beta_{1}x_{i,1} + \beta_{2}x_{i,2} + ... + \beta_{64}x_{i,64} + \sigma_{\epsilon_i}
$$

이 모델 안에 있는 65개의 알려지지 않은 모수들은 회귀 계수 $\beta = (\beta_{1}, ... , \beta_{64})$와 오차항의 표준편차인 $\sigma$의 벡터입니다. 파라미터 공간은 $\beta$에 대한 64차원 유클리디안 공간과 $\sigma$에 대한 양의 실수 직선입니다. 

### *사전 분포*

대부분의 상황에서, 65개의 모수에 대해 정확한 사전 믿음을 보여주는 결합 사전 확률 분포를 정의하는 것은 거의 불가능한 작업입니다. 그에 대한 대안으로 우리는 우리의 사전 믿음의 일부만을 나타내는 사전 분포를 사용할 것입니다. 우리가 보여주고 싶은 주된 믿음은 64개 설명 변수의 대부분이 당뇨병 진행에 큰 영향을 주지 못한다는 것입니다. 다른말로 표현하자면 대부분의 회귀 계수가 0일 것이라는 믿음입니다. 챕터 9에서 우리는 러프하게 각각의 회귀 계수가 50%의 사전 확률로 0과 같을 것이라는 믿음을 표현하는 $\beta$에 대한 사전 분포를 보여줄 것입니다.

### *사후 분포*

데이터 $\textbf{y} = (y_1, ..., y_{342})$와 $\textbf{X} = (x_1, ..., x_{342})$가 주어졌을 때, 사후 분포 $p(\beta | \textbf{y}, \textbf{X})$는 계산될 수 있고 각각의 회귀 계수 j에 대한 $Pr(\beta_{j} \neq 0 | \textbf{y}, \textbf{X})$를 구하는데 사용될 수 있습니다. 이 확률은 *그림 1.3.*의 첫 번째 패널과 같은 그래프로 그릴 수 있습니다. 64개 회귀 계수 각각이 50대 50의 0이 될 확률로 시작됐지만, 오직 6개의 $\beta_{j}$만이 0.5보다 크거나 같은 $\Pr(\beta_{j} \neq 0 | \textbf{y},\textbf{X})$를 가지게 됩니다. 대부분을 차지하는 나머지 회귀 계수들은 0이 될 높은 사후 확률을 가지고 있습니다. 이러하나 0값을 가지는 회귀 계수의 예상되는 수의 극적인 증가는 우선적으로 그러한 0의 값을 가지는 회귀 계수가 나오도록 하는것은 사전 분포이긴 하지만, 데이터에서 온 정보의 결과입니다.

![SmartSelectImage_2021-01-23-16-08-14](https://user-images.githubusercontent.com/57588650/105571683-6cb14800-5d95-11eb-855b-4b3234e77b1e.png)

**그림 1.3.** 각각의 회귀 계수가 0이 아닐 사후 확률들



### *예측 퍼포먼스와 베이지안이 아닌 방법론과의 비교*



우리는 이 모델이 얼마나 잘 작동하는지를 테스트 데이터를 추정하는데 이 모델을 사용함으로써 평가할 수 있습니다. $\hat{\beta}_{\text{Bayes}}$ 

$= \text{E}\[\beta | \text{y}, \text{X}\]$를 $\beta$의 사후 기댓값이라고 하고

$X_{test}$를 테스트 데이터셋에 있는 100명의 환자들의 데이터인 100X64 행렬이라고 합시다. 우리는 테스트 데이터셋에 있는 각각의 100개의 관찰값에 대한 추정된 값을 방정식 $\hat{y}_{test} = X \hat{\beta}_{\text{Bayes}}$를 활용해 계산할 수 있습니다.

그리고 추정된 값을 실제 관찰값인 $y_{test}$와 비교할 수 있습니다. $y_{test}$ 대 $\hat{y}_{test}$의 그래프는 그림 1.4.의 첫 번째 패널에 있고, 이는 $\hat{\beta}_{\text{Bayes}}$가 기본적인 변수들로부터 얼마나 당뇨병의 진행을 잘 예측하는지를 나타냅니다.

 이러한 $\beta$에 대한 베이지안 추정량을 어떻게 베이지안이 아닌 접근 방법과 비교할 수 있을까요? 가장 빈번하게 사용되는 회귀 계수 벡터에 대한 추정량은 대부분의 통계 소프트웨어 패키지에서 제공하는 *ordinary least squares*(OLS) 추정량입니다. $\beta$에 대한 OLS 회귀 계수 값 $\hat{\beta}_{ols}$는 관측된 데이터들의 잔차 제곱합(the sum of squares of the residuals(SSR))을 최소화하는 값입니다.

$$
\text{SSR}(\beta) = \Sigma^{n}_{i=1}(y_i - \beta^{T}x_i)^{2}
$$

그리고 $\hat{\beta}_{ols} = (X^{T} X)^{-1}X^{T} y$와 같은 식으로 주어집니다. 

 이 추정량을 기반으로 하는 테스트 데이터의 추정은 $X \hat{\beta}_{ols}$ 와 같이 계산되고 실제 관측된 데이터에 대해 그림 1.4.의 두 번째 패널과 같이 그래프로 그릴 수 있습니다.
 
 
 
 
 
 
 그리고 또한 $\hat{\beta}_{ols}$를 사용하는 것이 
 
 $\hat{\beta}_{Bayes}$ 보다 더 약한 관측값과 추정값과의 상관관계를 보인다는 점에 주목합시다. 
 
 
 
 
 
 
 
 
 
 이것은 각각의 예측값들에 대해 평균 제곱 예측 오차($\Sigma(y_{test,i} - \hat{y}_{test,i})^{2} / 100$)를 통해 정량적으로 나타낼 수 있습니다.
 




 
 
 
 
 
 
 OLS의 예측 오차는 0.67이고 이는 베이지안 추정량을 사용했을 때의 0.45보다 50%가량 더 높습니다. 이 문제에서 우리가 대충 정한 $\beta$에 대한 사전 분포가 오직 우리의 사전 믿음의 기본적인 구조(대부분의 상관계수가 0이될 확률이 높다는)만을 담고 있음에도 불구하고 이것은 OLS 추정량을 넘어서는 크게 향상된 예측 성능을 제공하는데 충분합니다. 

 이 예시에서 OLS 방식의 좋지 않은 성능을 가지는 이유는 정확하게 회귀 계수를 추정하기에는 너무 작은 샘플 사이즈를 가지고 있기 때문입니다. 그러한 상황에서 $\hat{\beta}_{ols}$로 계산된 데이터셋 안에 있는 $y$와 $X$ 값들 사이의 선형 관계는 자주 전체 모집단에서의 관계를 부정확하게 표현합니다.
 
 이 문제에 대한 일반적인 해결책은 몇몇 또는 많은 회귀 계수가 0으로 설정된 "희소(sparse)" 회귀 모델에 피팅하는 것입니다. 어떤 회귀 계수를 0으로 설정해야 하는지에 대한 한 가지 방법은 위에서 설명한 베이지안 접근법입니다. 또다른 유명한 방법은 Tibshirani(1996)에 의해 제안된 "라쏘(lasso)"이고 많은 사람들에게 열정적으로 연구되고 있습니다. 라쏘 추정량은 잔차제곱합의 수정된 버전인 $\text{SSR}(\beta : \lambda)$를 최소화 하는 $\beta$의 값 $\hat{\beta}_{lasso}$입니다.

 
$$
SSR(\beta : \lambda) = \sum^n_{i=1}(y_i - x_i^T \beta)^2 + \lambda \sum^p_{j=1} |\beta_j|
$$


![SmartSelectImage_2021-01-23-16-49-10](https://user-images.githubusercontent.com/57588650/105572518-f44d8580-5d9a-11eb-9273-913da348f90c.png)

**그림 1.4.** 베이즈 추정량을 사용했을 때(왼쪽 패널)와 OLS 추정량을 사용했을 떄(오른쪽 패널)의 관측값 대 추정된 당뇨병 진행 값

다르게 얘기하면, 라쏘 방식은 큰 값의 $|\beta_j|$에 패널티를 부과하는 것입니다. $\lambda$의 크기에 의존해 이 패널티는 $\hat{\beta}_{lasso}$의 몇몇 원소들을 0과 같게 만들 수 있습니다. 라쏘 방식이 베이지안이 아닌 개념에서 따와지고 연구되어왔지만, 사실 이것은 특정한 사전 분포를 사용하는 베이지안 추정량과 대응합니다. 라쏘 추정량은 각각의 $\beta_j$의 사전분포가 $\beta_j = 0$에서 날카로운 봉우리를 가지고 있는 확률 분포인 double-exponential 분포일 때 $\beta$의 사후 최빈값과 같습니다.

### *1.3* 우리는 어디로 가야하는가

위의 예시에서 나타낸 것 처럼, 베이지안 방법론의 사용처는 상당히 광범위합니다. 우리는 지금까지 베이지안 접근법이 어떻게 다음과 같은 것들을 제공하는지 알아보았습니다.

* 합리적이고 정량적인 학습 모델
* 크고 작은 샘플 크기에서 모두 작동하는 추정량
* 복잡한 문제들에 대한 통계적인 절차를 제공하는 방법들

베이지안 방법론의 장단점은 경험을 통해 이해할 수 있습니다. 다음 챕터에서 우리는 많은 수의 통계적인 모델들과 데이터 분석 예시들에 그들을 적용함으로써 이러한 방법론에 익숙해질 것입니다. 챕터 2에서 확률에 대해 복습한 후, 챕터 3과 4에서 우리는 베이지안 데이터 분석과 몇몇 간단한 일변수 통계 모델 개념을 컴퓨터로 구현하는 것을 배울 것입니다. 챕터 5,6,7는 정규 그리고 다변량 정규 모델들을 사용한 추론에 대해 논해볼 것입니다. 그 자체로도 중요하지만, 정규 모델들은 계층적 모델링(hierarchical modeling), 회귀 분석, 변수 선택과 혼합 효과 모형(mixed effects model)과 같은 더욱 복잡한 현대 통계 방법론들의 토대 또한 제공합니다. 이러한 고급 주제들과 그 외 다른 것들은 챕터 8부터 12에서 다룰 것입니다.

### *1.4. 시사점과 참고 자료*

알려지지 않았지만 결정된 값에 대한 불확실의 측정으로써의 확률이란 아이디어는 낡은 것입니다. 베이즈의 것을 포함한 중요한 역사적인 논문은 "An essay towards solving a Problem in the Doctrine of Chances" (Bayes, 1763)과 1814년에 출판되었고 현재는 Dover(Laplace, 1995)에 의해 출판되고 있는 라플라스의 "A Philosophical Essay on Probabilities"입니다.
 통계적 추론의 초기 의견들은 대부분 20세기에 논의되었습니다. 이 주제에 대한 대부분의 출판된 논문들은 한 쪽 또는 또다른 쪽을 차지하고 있고, 다른 쪽의 잘못된 성질을 포함하고 있습니다. 더 유용한 것은 다양한 관점의 통계학자들 사이의 논의입니다. Savage(1962)가 짧게 소개한 것을 Bartlett, Barnard, Cox, Pearson, Smith가 따라서 논의했습니다. Little(2006)은 베이지안과 빈도주의 통계 개념의 장단점을 생각했습니다. Efron(2005)는 간단하게 지난 2세기 동안의 서로 다른 통계적인 철학들의 역할과 미래 통계 과학에서의 베이지안과 베이지안이 아닌 방법론 사이의 상호작용에 대해 다뤘습니다.
