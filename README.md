# SVM tutorial
a simple tutorial for support vector machine

이 repository는 support vector machine을 처음  분들을 위해 작성되었습니다. 그래서 우선 여러 support vector machine에 대한 배경이론 제시, support vector machine의 개념 설명, 그리고 이를 sklearn의 모듈 구현체를 이용해 직접 실습해보는 순서로 구성하였습니다. 그리고 이 repository의 이론적인 토대는 첨부드린 논문과 고려대학교 강필성 교수님의 [유튜브 강의](https://www.youtube.com/watch?v=gzbafL28vA0&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW&index=8)를 참고하였음을 밝힙니다. 

## 목차
1. [Theoretical Background](#theoretical-background)
2. [Concepts of Support Vector Machine(SVM)](#concepts-of-support-vector-machinesvm)
3. [SVM Implementation](#svm-implementation)
4. [Conclusion](#conclusion)

---

## Theoretical Background
support vector machine(SVM)이 활발하게 쓰인 시점은 1998년 이후부터 2010년대 초반까지였습니다. 이후 딥러닝 계열의 알고리즘들이 SVM을 뛰어넘으면서 점차 SVM에 대한 열기가 사그라들었지만, 어쨌든 딥러닝 계열의 알고리즘들은 그 쓸모를 인정받기 위해 반드시 비교대상으로 넣어야 했던 알고리즘이 바로 SVM이었습니다. SVM이 활발하게 쓰일 당시 많은 머신러닝 알고리즘들은 shatter, VC dimension, structural/empirical risk minimization 등의 여러 이론적 배경을 기반하고 있는데요, 본격적인 SVM 소개에 앞서 이를 간단하게나마 먼저 짚고 넘어가보도록 하겠습니다. 

### Shatter
본론부터 말하자면 어떤 데이터셋 $S$가 있을 때, 어떤 함수 집합 $H$가 각 데이터 객체가 가질 수 있는 모든 이진분류 경우의 수(dichotomies)에 대해 이를 모두 성공적으로 분류할 수 있다면, $S$는 $H$에 의해 shatter된다( $S$ is shattered by $H$)고 합니다. 이게 무슨 소리인지 한번 예시를 들어보겠습니다. 

2차원 평면상에 겹치지 않고 한 직선위에 있지 않은 임의의 세 개의 점들의 집합 $S$와 함수의 집합 $H= \lbrace y\cdot sign(\vec{w}\vec{x} + b) | \vec{w} \ne \vec{0}, y = 1 or -1 \rbrace$가 있다고 합시다. 그러면 각 점이 가질 수 있는 이진분류의 경우의 수를 모두 표현하면 총 8개로, 아래 그림처럼 나타날 것입니다.(편의상, 각 이진분류의 label을 색으로 나타내어 붉은 색을 +1, 푸른색을 -1이라 하겠습니다.)
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/199213355-9a53c005-f269-4bab-afc7-d617929812a4.png" height="350px" width="800px"></p>

위 점들을 구분할 수 있는 직선 함수와 함께 위 그림을 다시 나타내면 아래와 같습니다.
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/199223090-a8ce9208-e823-4b77-95cc-ff31e5f2d610.png" height="350px" width="800px"></p>

보시는 바와 같이, $S$에서 만들어진 모든 이진분류 결과가 $H$의 function들에 의해 나타날 수 있음을 알 수 있으므로 $S$는 $H$에 의해 shatter 됨을 알 수 있습니다. 이러한 shatter의 개념에서 함수 집합이라는 말을 우리가 흔히 머신러닝에서 말하는 모델이라고 할 수 있을 것입니다. 결국 모델링도 어느 정도 정의되어있는 함수 집합에서 training error를 줄이는 함수를 찾아내는 것이기 때문입니다. 예를 들어 decision tree와 같은 경우도 $\vec{x}$를 $y$에 매핑하는 여러 decision tree 함수 중 특정 decision tree 함수를 찾아낸다고 볼수 있겠죠.

shatter의 개념을 곰곰히 생각해보면 결국 이것은 우리에게 어떤 이진분류 문제가 주어졌을 때, 우리가 100%의 accuracy의 분류 모델을 만들 수 있음이 확실한 데이터 객체의 수는 몇인지에 대한 개념과 일맥상통한다고 볼 수 있습니다. 예시로, 2차원 평면에서 아까와 같은 선형 분류 모델을 만들었을 때 모델이 shatter할 수 있는 최대 점의 개수를 생각해보면 3개임을 직관적으로 알 수 있습니다. 4개부터는 유명한 [XOR 문제와 같은 경우](https://i.stack.imgur.com/G7g23.png)가 존재하기 때문에 불가능하죠. 이러한 관점에서 접근한 것이 바로 VC dimension입니다.

### VC(Vapnik-Chervonenkis) Dimension
VC dimension은 한마디로 어떤 데이터셋 $S$와 함수집합 $H$에 의해 shatter 되는 $S$의 부분집합 중 가장 큰 부분집합의 크기입니다. 예를 들어 아까의 함수 집합 $H$와 같은 선형 분류기들의 $d$차원에서의 VC dimension은, 해당 집합의 함수들이 $d+1$개의 점을 shatter하는 경우가 존재하되 $d+2$개의 점부터는 shatter할 수 없기 때문입니다. 말이 좀 어려울 수 있는데, 예를 들면 2차원에서 선형분류기는 세개의 점을 shatter하는 경우가 존재합니다. 바로 위와 같은 경우가 그것이죠. 하지만 어떤 경우에도 2차원에 4개 이상의 점이 있을 경우 이를 shatter할 수 없습니다. 이는 사실 수학적으로 증명되어 있습니다만 우선 여기서는 생략하도록 하겠습니다(궁금하신 분들은 Mehryar Mohri의 Foundation of Machine Learning을 참조하시면 좋을 듯 합니다). 

VC dimension을 조금 더 직관적인 개념으로 생각해보면 이는 함수 집합 $H$(머신러닝에서는 모델)의 표현력(expressiveness power) 내지는 복잡도(complexity)로 생각할 수 있습니다. 쉽게 말해 더 많은 point를 shatter할 수 있을수록 해당 모델의 표현력이 높으며 동시에 복잡도도 높다는 것입니다. 예를 들어 앞서 제시한 XOR 문제를 가져와보겠습니다.
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/199238470-4b877836-c64b-4fff-acc5-41ab1bde6702.png" height="350px" width="800px"></p>

왼쪽 케이스의 경우는 선형 분류기에 의해 XOR 문제를 분류한 경우이고, 오른쪽은 비선형 분류기로 분류한 결과입니다. 믈론 수학적으로 엄밀하지는 않지만, 개념적으로 생각했을 때 오른쪽과 같은 비선형 분류기를 포함하는 함수집합이 존재한다면 이 함수집합의 2차원에서의 VC dimension이 선형분류모델의 VC dimension보다 크다고 할 수 있습니다. 그리고 보시다시피 비선형 분류기의 표현력, 복잡도 또한 선형 분류기에 비하여 크다는 것을 알 수 있습니다. 

따라서 어떤 모델의 VC dimension이 크다는 것은 어떤 training sample이 주어졌을 때, 이를 적합시키는(fit) decision boundary를 잘 만들어낼 수 있음을 의미합니다. 쉽게 말해 training error가 줄어드는 것이죠. 이를 다른 말로는 model의 capcity가 크다고도 표현합니다. 정리하면, 복잡한 모델일수록 ***1)*** VC dimension이 크고 ***2)*** 학습시 error가 줄어들기 쉽고 ***3)*** 모델의 capacity가 크다고 할 수 있겠습니다.

그런데 머신러닝을 공부하다 보면, training error가 줄어드는 것이 무조건 좋은게 아니라는 말을 한번쯤은 접하게 되곤 합니다. 바로 overfitting 문제가 그것입니다. 당연히 어느정도는 training error를 줄이도록 모델을 학습시켜야 하겠지만 training error를 완전히 줄여서 학습 데이터에 대한 decision boundary를 완벽하게 만들어낸다면 이 때문에 test set에서의 error는 오히려 올라가는 상황이 생기는 경우가 많습니다. 이러한 경우는 모델의 복잡도가 높은 경우 잘 발생할 수 있기 때문에 결국 모델의 복잡도가 올라갈수록(VC dimension이 커질수록) 모델의 test set에서의 error가 높아지는, 즉 일반화 성능(generalization ability)이 낮아질 가능성이 커지는 것이고, training error와 모델의 capacity 사이에는 trade-off가 있다고 할 수 있겠습니다. 이를 표로 정리하면 아래와 같습니다. 
| VC dimension | 모델의 복잡도 | 표현력     | capacity | training error | test error                      |
| :--:         | :--:          | :--:      | :--:      | :--:           | :--:                            |
| 작아지면     |  낮아진다.     | 낮아진다. | 작아진다. | 커진다.        |  (어느정도로 작아지느냐가 중요)  |
| 커지면       |  높아진다.     | 높아진다. | 커진다.   | 작아진다.      | 높아질 가능성이 커진다.          |

이러한 모델의 training error와 capacity 간의 trade-off 및 test error의 변화에 착안한 것이 바로 structural risk minimization(구조적 위험 최소화)입니다. 

### Structural Risk Minimization
앞서 설명했듯 모델의 capacity가 올라가면, 즉 VC dimension이 크면, training error는 줄어드는 경향을 보이고 test error는 커지는 경향을 보입니다. 이러한 관계를 아래 도표(K.R. Muller et al., An introduction to kernel-based learning algorithms, 2001)처럼 나타낼 수 있습니다. 
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/199259056-306d5c24-62df-4df3-907a-def79ff4be8e.png" height="350px" width="400px"></p>
위 그래프에서 confidence는 모델의 complexity에 따른 최대 complexity이고(uppper bound) empirical risk는 training error 입니다. 그리고 test error의 기댓값이 바로 expected risk이고 도표의 . 결국 위 도표에 따르면 test error의 기댓값은 최대 complexity와 training error 모두 적절한 수준인 경우에 최소화 할 수 있다는 것을 알 수 있습니다. 이러한 관계를 현대의 우리는 여러 사례를 통해 어느정도 이미 직관적으로 짐작하고 있지만, 이 도표가 만들어졌을즈음의 시기에는 위 도표와 같은 관계를 수학적으로 *증명*하기 위해 많은 노력을 기울였습니다. 해당 증명을 모두 다루지는 않겠지만, 중요한 부분만 간단하게 살펴보겠습니다. 

#### expected risk(test error) upper bound
데이터셋의 객체 $(\vec{x_i}, y_i)$들이 probability density function $P(\vec{x}, y)$에 의해 생성되었다고 합시다. 그리고 $x$가 n차원의 벡터이고, $y=1or-1$이라 할 때, 둘을 매핑하는 함수 $f:R^n \rightarrow \lbrace -1, +1 \rbrace$를 우리가 찾고자 하는 모델이라고 합시다. 함수 $f$의 parameter를 $w$라 하고, 실제 y와 f에 의해 매핑된 값의 차이를 input으로 하는 loss function을 $L$이라 하겠습니다. 그러면 우선 함수 $f$의 parameter $w$를 찾아내는 작업이 곧 training일 것입니다. 그리고 training이 되어 찾아낸 $w$에 대한 test error의 기댓값을 $R$이라 하면 아래와 같이 나타 낼 수 있겠습니다.

$$R(w) = \int L(y, f(x, w))dP(x, y)$$

우리의 목적은 저 기댓값을 최소화시키는 $w$를 찾는 것으로 생각할 수 있겠습니다. 그러나 우리가 $P(x, y)$를 알 수 있는 방법이 없다면 위 식을 적분하는 것이 불가능하기 때문에 sample $(\vec{x_i}, y_i)$을 $l$개 추출하여 emprical risk(즉, training error)을 이용하여 위 식을 추정할 것입니다. sample들에 대한 empirical risk는 아래와 같습니다.

$$R_{emp}(w) ={1 \over n} \sum_{i=1}^{l} L(y_i, f(x_i, w))$$

그리고 여기에 Vapnik 선생님에 의해 증명된 바(1995)에 따르면 $R(w)$는 $R_{emp}(w)$, 함수 $f$를 포함하는 함수공간 $F$의 VC dimension $h$, sample의 수 $l$, 그리고 특정 확률 값 $\delta$에 의해 $l>h$인 경우, $1-\delta$ 이상의 확률로 아래 부등식에 의해 bound 된다고 합니다.

$$R_(w)\leq R_{emp}(w)+\sqrt{\frac{h(ln({2l \over h})+1)-ln({\delta \over 4})}{l}}$$

위 식에서 empirical risk를 제외한 부분을 보면, $h$가 커질 때, $h(ln({2l \over h})+1)$도 커지므로 우리가 추정하고자 하는 함수 $f$가 포함된 집합의 VC dimension이 커지면 해당하는 제곱근 항 또한 커진다는 것을 알 수 있습니다. 반대로 $l$, 즉 sample 수가 많아지는 경우에는 제곱근항은 줄어든다는 것을 알 수 있죠. 이를 조금 더 직관적으로 보면 결국 모델의 complexity, capcity, VC dimension이 크면 일반화 성능이 떨어질 가능성이 커지고, 반대로 표본의 수가 많아지면 일반화 성능이 올라갈 가능성이 커진다는 것이죠. 그리고 당연히 empirical error, 즉 training error가 크면 일반화 성능이 낮을 가능성이 큰 것이고요. 

더불어 Vapnik 선생님이 증명하신 바에 따르면, $l/h$가 크면, 즉 모델의 복잡도에 비해 sample이 충분히 크면 결국 위 부등식의 우변은 empirical risk에 근사하게 됩니다. 때문에 이러한 상황에서는 empirical risk를 minimization하는 것에만 신경써도 일반화 성능이 커질 가능성이 여전히 높을 수 있습니다. 이러한 접근법을 empirical risk minimization이라고 합니다. 말은 어렵지만 결국 그냥 training에만 신경쓰고 overfitting은 신경 안쓰겠다는 소리인거죠. 이런 알고리즘이 있나 싶긴하지만 사실 대부분의 알고리즘이 이러합니다. 엄밀히 말하자면 $l/h$가 큰지 안큰지 신경안쓰고 그냥 학습성능 최대화에 신경을 쓰는 경우가 많습니다. 우리가 흔히 알고있는 logistic regression, decision tree, neural network 계열의 알고리즘들이 사실 학습 성능의 최대화를 하도록 설계되어있고, 때문에 과적합 방지를 위해 regularization이나 dropout, early stopping 등의 다양한 방법을 활용하는 것입니다. 

반대로 모델의 복잡도가 크거나 표본이 작은 경우($l/h$가 작은 경우) 모델의 복잡도가 커지면 일반화 성능이 낮아질 수 있기 때문에 모델 복잡도를 어느정도 제한하도록 함으로써 최적의 일반화 성능을 가지는 모델을 찾아내려 하는데, 이러한 접근법이 바로 structural risk minimization인 것입니다. 그리고 앞서 제시한 바와 같이 많은 모델들은 기본적으로 ERM만 수행하지만 여기에 regularization 등의 테크닉을 적용하여 결과적으로 structural risk minimization을 수행하는 방향으로 머신러닝 모델링이 진행되는 경우가 굉장히 많습니다. 그리고 어떤 머신러닝 알고리즘의 경우에는 애초에 학습과정 자체에 structural risk minimization이 포함된 것도 있습니다. 바로 support vector machine이 그러한 예시입니다. 

추후 다시 설명하겠지만, SVM은 margin을 최대화하는 초평면을 찾는 알고리즘입니다. 이때 margin을 $p$, SVM이 찾아내는 hyperplane의 VC dimension을 $h$, 데이터셋을 모두 포함하는 최소 초구(hypersphere)의 반지름을 $R$, 그리고 데이터셋의 차원 $D$에 대해 아래와 같은 관계가 성립합니다.

$$h \le min (\lceil R^2/p^2 \rceil, D)+1$$

보시다시피 결국 margin $p$가 증가하면 $min (\lceil R^2/p^2 \rceil, D)$이 $\lceil R^2/p^2 \rceil$와 같고 동시에 해당 값이 줄어들것이고, 이는 결국 $h$가 감소하는 결과를 가져오며, SRM의 수식에 따르면 $h$의 감소는 곧 expected risk, 즉 test error가 줄어들 가능성을 높이는 것입니다. empirical risk만을 줄이는 것이 아니라 전체적인 capcity 및 empirical risk를 모두 고려하는 것이죠. 따라서 SVM은 곧 SRM을 수행하는 알고리즘이라고 볼 수 있겠습니다.

여기까지 SVM이 지향하는 바를 이해하기 위한 배경이론을 살펴보았습니다. 이제부터 support vector machine의 개념을 살펴보겠습니다.

---

## Concepts of Support Vector Machine(SVM)
SVM을 한마디로 정의하면 "선형분류기"입니다. 데이터를 +1과 -1로(사실 다른 수여도 괜찮습니다.)이진분류하는 초평면(hyperplane)을 찾아내는 알고리즘들 중 하나이죠. 모든 종류의 초평면은 $w \cdot x + b =0$ 로 나타낼 수 있으니까($x$는 미지수 벡터 $w$는 가중치 벡터입니다.) SVM은 결국 $w \cdot x + b =0$에서 $w$랑 $b$를 찾아내는 것인 셈입니다. 그럼 그것을 어떻게 찾아낼까요? 우선 가장 기본적인 형태의 SVM의 경우를 살펴보겠습니다.

### SVM for Linearly Seperable Case(hard margin)
가장 기본적인 형태의 SVM의 case를 linearly seperable한 case에서의 SVM, 혹은 linear hard margin SVM이라고 합니다. 쉽게 말해 데이터가 선형분류기로 완벽하게 분류될 수 있는 상황에서의 SVM이죠. 일단 데이터를 분류하는 초평면 $w \cdot x + b =0$가 있다고 할 때, 이것에 평행하면서 동일한 간격으로 떨어진 두 개의 초평면 $w \cdot x + b =1$과 $w \cdot x + b =-1$을 생각해봅시다. 그리고 만약 데이터의 이진분류가 초평면에 의해 완벽하게 가능하다면, $w \cdot x + b \ge 1$의 영역에는 +1로 labeling된 객체, $w \cdot x + b \le -1$의 영역에는 -1 labeling된 객체만 존재하도록 $w \cdot x + b =0$를 적절히 조절할 수도 있겠죠. 그림으로 나타내면 아래와 같은 상황이 될 것입니다. 
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/199650392-215e2bd5-7b92-4fd0-b02d-ff2a44acbb65.png" height="450px" width="600px"></p>

위와 같은 상황에서, SVM에서 찾고자 하는 초평면은 $w \cdot x + b =1$과 $w \cdot x + b =-1$ 사이의 "간격"을 최대화하는 $w \cdot x + b =0$입니다. 이 간격을 margin이라고 부릅니다. 그러면 이 간격을 최대화하기 위해서는 무엇을 해야할까요? 초평면 $w \cdot x + b =0$를 찾아내는 과정은 결국 $w$와 $b$를 찾아내는 과정인데, margin은 이 중 $w$에만 관련이 있습니다. 왜 그러한지 한번 직접 margin을 구해보면서 알아보겠습니다.

우선 평행하는 평면간의 간격은 두 평면 모두에 수직인 직선과 평면 둘의 교점 사이의 거리라고 할 수 있겠습니다. 우리가 간격을 계산하고자 하는 두 평면 $w \cdot x + b =1$과 $w \cdot x + b =-1$에 모두 수직인 직선의 방향벡터는 곧 두 평면의 법선 벡터, 즉 $w$와 동일합니다. margin을 $p$라 하고, $w \cdot x + b =1$과 $w \cdot x + b =-1$ 위의 점을 각각 $x_+$, $x_-$라 하면,

$w \cdot x_+ = 1-b$, $w \cdot x_- = -1-b$이므로 $w \cdot (x_+ - x_-)=2$ 이고

$w \cdot (x_+ - x_-)= |w| \cdot |(x_+ - x_-)|$ 인데, 앞서 정의했듯 $|(x_+ - x_-)| = margin = p$이므로

$|w| \cdot |(x_+ - x_-)|=2 \Rightarrow margin=p=\frac{2}{|w|} $입니다. 

**따라서, margin은 $w$의 함수입니다!**

정리하면, SVM은 $w \cdot x + b \ge 1$의 영역에는 +1로 labeling된 객체, $w \cdot x + b \le -1$의 영역에는 -1 labeling된 객체만 존재하도록 한다는 조건을 만족하면서 margin, 즉 $2/|w|$를 최대화하는 초평면 $w \cdot x + b =0$을 찾아내는 알고리즘이라고 할 수 있겠습니다. 다만 하나만 첨언하자면 실제로는 $2/|w|$의 분모에 제곱근이 포함되어 있기 때문에, 이를 동일한 의미를 가지도록 $|w|^2/2$를 최"소"화하도록 학습을 진행합니다. 이는 결국 $|w|^2/2$을 목적함수로 하고 $w \cdot x + b \ge 1$의 영역에는 +1로 labeling된 객체, $w \cdot x + b \le -1$의 영역에는 -1 labeling된 객체만 존재하도록 한다는 것을 제약조건으로 가지는 최적화 문제로 formulation 할 수 있습니다.

### SVM for Linearly Non-Seperable Case(soft margin)
지금까지 살펴본 바와는 다르게 데이터를 완벽하게 분류할 수 있는 초평면을 찾을 수 없는 경우에서의 SVM은 linearly non-separable case의 SVM 혹은 linear soft margin SVM이라고 합니다. 앞서 hard margin case와 soft margin case와는 무엇이 다를까요? soft margin case에서는 완벽하게 분류되지 않는 점들에 대한 penenalty term인 $\xi$를 도입하여 linear hard margin case의 조건을 변형하고 $|w|^2/2$가 아닌 $|w|^2/2 + C \sum \xi$를 최소화하는 알고리즘이 됩니다. 그러한 케이스의 예시는 아래와 같이 나타낼 수 있습니다. 
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/199658398-fb7678b1-2023-4457-af99-88f4ed866bdb.png" height="450px" width="630px"></p>

이러한 케이스의 SVM 또한 $|w|^2/2 + C \sum \xi$를 목적함수로 가지고, linear hard margin case의 조건이 변형된 형태의 제약조건을 가지는 최적화문제로 formulation 됩니다. 우선 제약조건 부분에 대해 조금 더 자세히 살펴보겠습니다.

Soft margin SVM에서는 데이터셋에서 +1로 labeling된 객체가 $w \cdot x + b \ge 1$의 영역에 있지 않는 경우를 허용하되 $w \cdot x + b \ge 1$의 영역에 있지 않은 객체에는 그 객체와 $w \cdot x + b = 1$ 사이의 거리를 $\xi$라 하여 $\xi$만큼의 penalty를 부과합니다.(때문에 영역내에 들어와 있는 객체들에는 penalty $\xi$가 0인 것으로 생각할 수 있을 것입니다) 이러한 제약조건은 결국 +1로 labeling된 객체가 $w \cdot x + b \ge 1-\xi$의 영역에 포함되어야 한다는 조건과 같게 됩니다. 그리고 같은 원리로 -1로 labeling 된 객체에 해당하는 제약조건은 $w \cdot x + b \le -1+\xi$가 되겠죠. 정리하면, soft margin SVM에서의 최적화 문제의 목적함수는 $|w|^2/2 + C \sum \xi$, 제약조건은 +1로 labeling된 객체의 경우 $w \cdot x + b \ge 1-\xi$를 만족해야하고, -1로 labeling 된 객체의 경우 $w \cdot x + b \le -1+\xi$를 만족해야한다는 것으로 요약할 수 있겠습니다. 

이제 그럼 목적함수 $|w|^2/2 + C \sum \xi$를 한번 살펴보겠습니다. 일단 저 목적함수를 최소화하게되면 margin은 커지게 하면서 penalty term $\sum \xi$은 작게 만들어 준다는 것이 직관적으로 이해되실겁니다. 그리고 한발짝 더 나아가서 생각해보면, margin과 penalty term의 사이의 trade off 관계가 있음을 알 수 있습니다. penalty를 최대한 줄인다는 것은 데이터 객체들을 penalty가 부과되지 않는 영역으로 최대한 많이 옮겨 주어야 한다는 것입니다. 그러려면 당연히 margin을 측정하는 두 초평면 $w \cdot x + b =1$과 $w \cdot x + b =-1$ 사이의 공간을 줄여야 할 것입니다. 아래의 그림이 바로 그 예시로, 같은 데이터셋에 대해 오른쪽 그림처럼 margin이 줄어들어야 penalty가 작음을 알 수 있습니다.  
<p align="center"><img src="https://user-images.githubusercontent.com/112034941/199690481-18d03fae-4386-4eec-9e2c-67ff1169904a.png" height="400px" width="1000px"></p>

근데 여기서 들만한 궁금증은 $\sum \xi$ 앞의 $C$는 갑자기 뭐냐는 것입니다. 이것은 hyper parameter, 즉 알고리즘의 사용자가 직접 정해줘야 하는 값입니다. 원하는 대로 아무거나 정할 수 있는데요, 이것이 가지는 의미는 꽤나 직관적입니다. 목적함수 $|w|^2/2 + C \sum \xi$를 최소화하고 싶은데, $C$가 커지면 어떻게 될까요? 최적화 했을때, $\sum \xi$가 더 작아질 수 있도록 하는 방향으로 최적화가 이뤄질 것이고, 결국 margin은 줄어들 것입니다. 반대로 $C$가 작아지면 $\sum \xi$가 조금 더 커져도 될 것이고 대신 margin이 커지게 되겠죠. 위의 그림에서 왼쪽은 $C$가 더 작은 경우이고, 오른쪽은 $C$가 더 큰 경우로 보아도 무방하겠습니다.  

여기까지 soft margin case의 SVM의 개념을 살펴보았습니다. 그러면 이게 SVM의 최종진화 버젼일까요? 놀랍게도 그렇지 않습니다. hard margin에서 soft margin으로 넘어올 때, 우리는 데이터셋의 이진분류를 완벽하게 수행할 수 있는 선형 경계면이 없는 경우, 즉 linearly seperable하지 않은 경우에도 SVM을 수행하려면 어떻게 해야 하는가에 대해 살펴보았습니다. 다음으로 살펴볼 SVM은 이에 더해 linear한 분류경계면을 찾아내지 않는 SVM을 한번 살펴보겠습니다.

### SVM for Non-Linearly Non-Seperable Case(nonlinear soft margin)
지금까지 살펴본 SVM은 선형분류경계면을 찾아내는 알고리즘이었습니다. 그러나 이번에 살펴볼 SVM은 nonlinear soft margin SVM 내지는 kernel SVM으로, 이름에서 보시다시피 "비"선형 분류경계면을 찾아내는 알고리즘입니다. 새삼스래 혼란스럽네요. 분명 SVM은 처음에 선형분류경계면을 찾는다고 했었는데, 이제 와서 갑자기 비선형 분류경계면을 찾는다니...... 조금 더 혼란스러우실 만한 말씀을 드리자면, 지금 설명 드리는 SVM 알고리즘도 선형분류경계면을 찾는 알고리즘입니다. 그러니까 선형분류경계면을 찾으면서 비선형 분류경계면을 찾는 것이죠. 이 모순은 분류경계면을 찾는 차원을 명시해서 설명드리면 해결됩니다. 이렇게 말이죠.

**Nonlinear soft margin SVM은 고차원으로 mapping된 데이터에 soft margin SVM을 수행하여 선형 분류경계면을 찾는 알고리즘으로, 여기서 찾은 고차원의 선형 분류경계면은 저차원 공간에서의 비선형분류경계면이 됩니다.** 아래의 예시를 보시죠.

<p align="center"><img src="https://user-images.githubusercontent.com/112034941/199701644-a0b132b4-08a0-4fa2-908f-58629bf23841.png" height="450px" width="1000px"></p>

왼쪽 그림처럼 2차원 공간의 데이터셋을 보면 아무래도 선형경계면보다는 원과 같은 비선형 경계면을 찾아내는 것이 더 좋아보입니다. 이때 kernel SVM은 왼쪽그림과 같은 2차원 데이터를 3차원으로 mapping하여 새로운 $z$축의 값을 $z=x^2+y^2$로 하도록 하여 새로운 3차원 공간에서의 선형분류경계면을 찾아냅니다. 이는 보시다시피 2차원에서 비선형분류경계면을 찾아내는 것과 동일하죠. 수식적으로도 한번 살펴보면, linear soft margin case와 대부분이 동일하되, 제약 조건이 달라집니다. 앞서 제시했듯이, 고차원에서의 mapping된 데이터에 soft margin SVM을 수행하는 것이기 때문에 이 mapping함수를 $\Phi (x)$라 하면, nonlinear case의 SVM의 제약조건은 +1로 labeling된 객체의 경우 $w \cdot \Phi (x) + b \ge 1-\xi$를 만족해야하고, -1로 labeling 된 객체의 경우 $w \cdot \Phi (x) + b \le -1+\xi$를 만족해야한다는 것으로 정리할 수 있습니다. 앞서 살펴보았던 제약조건의 $x$ 부분이 그냥 $\Phi (x)$로 바뀐 것이죠. 

이때, 아주 아주 중요한 개념이 하나 등장합니다. 바로 kernel입니다. 굉장히 유명하고 중요한 개념이고, 사실 SVM이 성공적인 알고리즘이 된건 바로 이 kernel 덕이라고 해도 과언이 아닙니다. 지금까지 설명 속에서 우리는 총 세 가지case의 SVM의 개념을 살펴보았고, 각 경우마다 어떤 식으로 문제를 formulation 해야하는지도 함께 살펴보았습니다. 하지만 그걸 그래서 어떻게 푸는지는 살펴보지 않았죠. 이 과정은 라그랑지안 승수법(lagrangian multiplier method), primal-dual interior point method를 활용하여 해결해야 하는 문제입니다만 그 부분은 추후에 기회가 된다면 다뤄보기로 하고, 지금은 kernel과 연관된 부분의 개념만 보겠습니다. 

Non-linear soft margin case의 최적화 문제를 풀다보면 문제를 primal 문제에서 dual 문제로 전환하게 되고 결국 dual 문제를 해결하려면 우리가 알아야 하는 것이 각 데이터 객체 별 고차원으로의 mapping된 값의 "내적값" 뿐이라는 것을 알게됩니다. 수식으로 나타내자면 $\Phi (x_i) \cdot \Phi (x_j)$죠. 데이터 객체 $x_i,x_j$를 input으로 받아 $\Phi (x_i) \cdot \Phi (x_j)$를 output으로 하는 함수가 있다면, 우리는 굳이 데이터를 먼저 고차원으로 mapping하는 함수 $\Phi$를 찾아주어야 할 필요가 없을 것입니다. 여기가 바로 kernel이 사용되는 순갑입니다. kernel은 정확하게 $x_i,x_j$를 input으로 받아 $\Phi (x_i) \cdot \Phi (x_j)$를 내보내 주는 함수 $K(x_i,x_j)$를 말합니다. 그래서 그냥 kernel 함수만 정해주면 데이터 객체의 고차원으로의 추가적인 mapping 후 내적하는 과정없이 kernel에서의 input과 output만 수행해주면 되는 것입니다. 

이러한 kernel 함수의 종류는 다양합니다. RBF(Gaussian) kernel, polynomial kernel, sigmoid kernel 등 다양한 kernel이 존재하지만, 이 중 가장 자주 쓰이는 것은 RBF kernel로, 수식은 $K(x_i,x_j)=exp(-\gamma|x_i-x_j|^2)$, $where \gamma=1/\sigma ^2$와 같습니다. RBF kernel에서 추가로 알아두면 좋은 점은 $\gamma$가 커지면(혹은 $\sigma$가 작아지면) 더 비선형성이 높은 분류경계면이 생겨난다는 점입니다. 이를 최적화 과정에서 kernel 함수의 역할을 생각해보면 결국 라그랑지안 승수가 마구잡이로 커지는 것에 도움을 주어 다양한 support vector가 나오게 하기 때문이라고 해석할 수도 있겠으나 이에 관해 다루지 않았으니 그냥 이 정도로 하고 넘어가도록 하겠습니다.

여기까지, support vector machine의 개념에 대해 살펴보았습니다. 이러한 과정을 통해 support vector machine와 관련한 이론에 대해서 배운 내용을 파이썬의 모듈들을 이용해 실제로 실험해보도록 하겠습니다.

---

## SVM Implementation
우선 실험에 앞서 필요한 모듈 등의 버젼은 아래와 같습니다.
| env_name   | version |
|------------|---------|
| python     | 3.8.3   |
| numpy      | 1.19.2  |
| matplotlib | 3.5.2   |
| pandas     | 1.4.3   |
| sklearn    | 1.1.1   |

사용할 데이터셋은 kaggle의 [heart attack dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)입니다. 파일이 두 개가 있는데, haert.csv를 사용했습니다. 그리고 SVM의 구현을 위해 사용해볼 모듈은 sklearn.svm의 SVC 입니다. 실험해볼 내용은 총 3가지로, 아래와 같습니다.
1. $l/h$, 즉, 함수의 복잡도에 따른 test set에서의 성능, training set에서의 성능 비교 분석
2. 비선형적 mapping 상황에서의 RBF kernel SVM과 그냥 linear soft margin SVM의 성능 비교.
3. 비선형적 mapping 상황에서의 RBF kernel SVM의 gamma와 C에 따른 분류 경계면 비교.

### 함수의 복잡도에 따른 test error 및 training error의 성능 비교 분석
우선 1.을 시험해보기 위해 다음과 같이 k-fold 함수와 evaluation 함수를 구현하였습니다. 5-fold로 실험을 진행하였고, evaluation metric은 accuracy, f1-score를 사용하였습니다. 비교대상으로 그냥 linear soft margin SVM과 그보다 복잡도가 높은 RBF kernel SVM 모델(with C=1000)을 골라 실험을 진행하였습니다. 비교 분석의 포인트는 총 두가지로, 첫째는 복잡도가 높아질 경우 test 성능, 즉 일반화 성능이 떨어지는지를 살펴보고 두번째로는 복잡도가 낮아지는 경우 실제로 empirical error, 즉 training set에서의 성능이 일반화 성능에 비슷한지를 살펴보았습니다. 

1. k-fold 함수: 각 fold에 0과 1의 비율이 유사하게 구성되도록 고려함.
```python
def kfold(data, fold, seed):
    import random
    idx_set_1 = data.loc[data.output== 1].index.tolist()
    idx_set_0 = data.loc[data.output == 0].index.tolist()
    size_1 = round(len(idx_set_1)/fold)
    size_0 = round(len(idx_set_0)/fold)
    folded_idx_set = []
    for i in range(fold):
        if (i == fold-1):
            folded_idx_set.append(idx_set_1+idx_set_0)
        else:
            random.seed(seed)   
            folded_idx_set.append(list(random.sample(idx_set_1, size_1))+list(random.sample(idx_set_0, size_0)))
            idx_set_1 = list(set(idx_set_1)-set(folded_idx_set[len(folded_idx_set)-1]))
            idx_set_0 = list(set(idx_set_0)-set(folded_idx_set[len(folded_idx_set)-1]))
    
    return folded_idx_set

kfold_idx_set = kfold(data=data, fold=5, seed=1103)

trn_X = []
trn_y = []
tst_X = []
tst_y = []
index_set = data.index.tolist()

for i in range(5):
    total_idx = set(index_set)
    trn_X.append(data.loc[list(total_idx-set(kfold_idx_set[i]))][x_features])
    trn_y.append(data.loc[list(total_idx-set(kfold_idx_set[i]))]['output'])
    tst_X.append(data.loc[kfold_idx_set[i]][x_features])
    tst_y.append(data.loc[kfold_idx_set[i]]['output'])
```

2. 성능평가를 위한 evaluation 함수
```python
from sklearn.metrics import confusion_matrix
def evaluation(fold_num, model, trn_X, trn_y, tst_X, tst_y, data):
    y_pred = model.predict(trn_X)
    tn, fp, fn, tp = confusion_matrix(trn_y, y_pred).ravel()
    data.loc[fold_num, 'trn_accuracy'] = (tp+tn)/(tn+fp+fn+tp)
    data.loc[fold_num, 'trn_f1'] = 2*tp/(fp+2*tp+fn)
    

    y_pred = model.predict(tst_X)
    tn, fp, fn, tp = confusion_matrix(tst_y, y_pred).ravel()
    data.loc[fold_num, 'tst_accuracy'] = (tp+tn)/(tn+fp+fn+tp)
    data.loc[fold_num, 'tst_f1'] = 2*tp/(fp+2*tp+fn)

    return data
```

3. 실험 코드
```python
eval_data = pd.DataFrame([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], columns=['trn_accuracy', 'trn_f1', 'tst_accuracy', 'tst_f1'])
from sklearn.svm import SVC
# simple model: linear soft margin
for i in range(5):
    fold_num = i
    model = SVC(kernel='linear', C=1)
    model.fit(trn_X[fold_num], trn_y[fold_num])
    eval_data = evaluation(fold_num, model, trn_X[fold_num], trn_y[fold_num], tst_X[fold_num], tst_y[fold_num], eval_data)

eval_data.mean()
print(100*abs(eval_data.mean()['trn_accuracy']-eval_data.mean()['tst_accuracy'])/eval_data.mean()['tst_accuracy'])
print(100*abs(eval_data.mean()['trn_f1']-eval_data.mean()['tst_f1'])/eval_data.mean()['tst_f1'])

# complex model: nonlinear soft margin
for i in range(5):
    fold_num = i
    model = SVC(kernel='rbf', C=1000)
    model.fit(trn_X[fold_num], trn_y[fold_num])
    eval_data = evaluation(fold_num, model, trn_X[fold_num], trn_y[fold_num], tst_X[fold_num], tst_y[fold_num], eval_data)

eval_data.mean()
print(100*abs(eval_data.mean()['trn_accuracy']-eval_data.mean()['tst_accuracy'])/eval_data.mean()['tst_accuracy'])
print(100*abs(eval_data.mean()['trn_f1']-eval_data.mean()['tst_f1'])/eval_data.mean()['tst_f1'])
```

결과는 아래와 같습니다.
![image](https://user-images.githubusercontent.com/112034941/199727492-9488c453-03cd-4d98-ad41-7e7a3ee3cee5.png)

우선 보시다시피, 전체적 성능이 복잡도가 작은 모델인 linear soft margin SVM 모델에서 높은 것을 확인하실 수 있고, 실제로 training set과 test set 간의 성능 차이도 복잡도가 더 작은 모델에서 더 작게 나타났음을 확인하실 수 있습니다. 앞서 theoretical background의 SRM 부분에서 설명드렸던 양상이 그대로 나타난 것이죠. 특히 이 데이터셋의 경우 sample의 수가 작기 때문에 복잡도에 의한 성능 저하가 더 뚜렷하게 나타난 것으로 보입니다. 다음 실험으로 넘어가겠습니다.

### 비선형적 mapping 상황에서의 RBF kernel SVM과 그냥 linear soft margin SVM의 성능 비교
우선 현재 실험하는 데이터셋을 t-SNE를 통해 비선형적으로 2차원에 mapping해보겠습니다. 실험 코드와 결과는 아래와 같습니다.
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore") # warning 무시

tsne = TSNE(n_components=2, random_state = 1103)
tsne_data = tsne.fit_transform(data[x_features])
tsne_data = pd.DataFrame(tsne_data, columns=['z1', 'z2'])
plt.figure(figsize=(10,10))
plt.title('nonlinear mapping by sklearn_TSNE')
plt.scatter(tsne_data.z1, tsne_data.z2, c=data.output, alpha=0.7)
```
![image](https://user-images.githubusercontent.com/112034941/199740145-e6a49a5a-bedc-4ad2-92ef-5da351d32b9f.png)

결과를 보면 매우 비선형적으로 mapping되어 있음을 확인할 수 있습니다. 이를 1.에서와 동일하게 linear soft margin SVM과 RBF kernel SVM을 활용하여 성능을 비교해보겠습니다.
```python
tsne_data = pd.concat([tsne_data, data.output], axis=1)
tsne_data.columns = ['z1', 'z2', 'output']
kfold_idx_set = kfold(data=tsne_data, fold=5, seed=1103)

trn_X = []
trn_y = []
tst_X = []
tst_y = []
index_set = data.index.tolist()

for i in range(5):
    total_idx = set(index_set)
    trn_X.append(tsne_data.loc[list(total_idx-set(kfold_idx_set[i]))][['z1', 'z2']])
    trn_y.append(tsne_data.loc[list(total_idx-set(kfold_idx_set[i]))]['output'])
    tst_X.append(tsne_data.loc[kfold_idx_set[i]][['z1', 'z2']])
    tst_y.append(tsne_data.loc[kfold_idx_set[i]]['output'])

eval_data = pd.DataFrame([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], columns=['trn_accuracy', 'trn_f1', 'tst_accuracy', 'tst_f1'])

# simple model: linear soft margin
for i in range(5):
    fold_num = i
    model = SVC(kernel='linear', C=1)
    model.fit(trn_X[fold_num], trn_y[fold_num])
    eval_data = evaluation(fold_num, model, trn_X[fold_num], trn_y[fold_num], tst_X[fold_num], tst_y[fold_num], eval_data)

print(eval_data.mean())

# complex model: nonlinear soft margin
for i in range(5):
    fold_num = i
    model = SVC(kernel='rbf', C=1000)
    model.fit(trn_X[fold_num], trn_y[fold_num])
    eval_data = evaluation(fold_num, model, trn_X[fold_num], trn_y[fold_num], tst_X[fold_num], tst_y[fold_num], eval_data)
    
print(eval_data.mean())

```
![image](https://user-images.githubusercontent.com/112034941/199733110-2fc4f8a4-e983-4e36-9c9f-f5361fcb5229.png)

결과를 보면 training dataset에서의 성능은 확실히 RBF kernel을 사용한 모델에서 더 뛰어남을 알 수 있습니다. 다만 test set에서의 성능은 서로 유사한데요, training set에서의 성능과 일반화 성능이 서로 trade off 관계에 있으며 이는 모델의 복잡도에 의해 조절될 수 있음을 상기해보면 아마도 우리 모델은 복잡도가 높아 empirical error는 낮아졌으나 test error는 조금 높아진 상황이라고 추측할 수 있을 것입니다. 따라서 모델 성능을 높이기 위한 해결책으로써 모델의 복잡도를 낮추기 위해 C를 10으로 감소시켰습니다. 결과는 아래와 같습니다.
![image](https://user-images.githubusercontent.com/112034941/199735199-0843e2eb-e375-4fef-9b22-678a81070601.png)

결과를 보면 성공적으로 일반화 성능이 상승하였음을 확인할 수 있습니다. 마지막 실험으로 넘어가겠습니다.

###  비선형적 mapping 상황에서의 RBF kernel SVM의 gamma와 C에 따른 분류 경계면 비교
앞서 만들어낸 t-SNE에 의한 비선형적 mapping data를 RBF kernel SVM을 이용하여 분류해보고, $\gamma$(gamma)와 C를 조절하여 그 변화를 확인해보겠습니다. 우선 분류경계면을 그리기 위한 함수는 아래와 같습니다. [이 링크](https://ekfkdlxm.tistory.com/41)의 코드를 참조하였습니다.
```python
def svm_plot(tsne_data, model):
    import numpy as np
    import matplotlib as plt
    plt.figure(figsize=(15,15))
    plt.scatter(tsne_data.z1, tsne_data.z2, c=tsne_data.output, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)# -1 margin, decision boundary
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape) # +1 margin

    #margins,decision boundary
    ax.contour(XX, YY, Z, colors=['b','r','b'], levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])

    plt.show()
```

그리고 실험은 다음과 같은 코드에서 C와 $\gamma$를 변경해가며 진행하였습니다. 실험한 C는 1, 10, 50, 100, 1000, 10000이고 C를 10으로 고정시킨 상태에서 실험한 $\gamma$는 0.0001, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 1입니다. 
```python
C = 10
gamma = 0.0001
model = SVC(kernel='rbf', C=C, gamma=gamma)
model.fit(tsne_data[['z1', 'z2']], tsne_data.output)

svm_plot(tsne_data, model)
```
우선 C의 변화에 따른 분류경계면의 변화는 아래와 같습니다. 붉은 실선이 분류 경계면, 푸른 점선이 margin의 경계면입니다. 
![image](https://user-images.githubusercontent.com/112034941/199748979-1aeda47e-3215-402c-a081-a45cca481cdf.png)

앞서 개념에서 C가 증가하면 결국 margin을 줄이게 된다는 것을 언급한 적이 있습니다. 그리고 실제 결과를 보면 C의 증가에 따라 실제로 margin이 줄어든 형태의 분류경계면이 나타나고 있음을 알 수 있습니다. 그리고 더불어 비선형성 또한 높아지고 있는 것을 알 수 있습니다.

다음은 $\gamma$의 변화에 따른 분류경계면의 변화입니다. 
![image](https://user-images.githubusercontent.com/112034941/199751424-860208e2-fe00-45a2-85a6-fc07893cb4a5.png)

앞서 개념 설명에서와 같이 $\gamma$가 증가하면 비선형성이 증가한다는 것을 언급한 적이 있습니다. 그리고 실제 결과를 보면 $\gamma$의 증가에 따라 비선형적인 분류경계면의 양상이 나타나고 있음을 확인할 수 있습니다. $\gamma$와 C의 영향력을 그림을 통해 비교해보자면 비슷한 비율로 증감했을때, $\gamma$의 영향이 C의 영향에 비해 매우 큰 것으로 보입니다. 왜냐하면 C를 1에서 10000으로 증가시켰을 때보다 $\gamma$를 0.0001에서 1로 증가시켰을 때 훨씬 극단적인 변화가 나타나기 때문입니다. 이러한 현상의 이유를 생각해보자면 앞서 설명한적은 없으나 우리가 kernel SVM의 최적화 문제를 풀다보면 아래의 실제 식과 같아짐을 확인할 수 있습니다.(아래 그림은 고려대학교 강필성 교수님의 강의자료를 참조하였습니다.) 
![image](https://user-images.githubusercontent.com/112034941/199753273-acaf7d44-02e0-4deb-8fa7-d016155d646e.png)

C가 커지면 라그랑지안 승수의 범위가 늘어나는 것에 그치는 반면 $\gamma$가 커지면 목적함수에서의 음수항을 거의 0으로 만들어주는 역할을 하게 되는데, 직관적으로 생각해보았을 때 $\gamma$의 변화에 라그랑지안 승수가 더 민감하게 반응할 것이며, 결국 이 라그랑지안 승수가 support vector와 연관이 있고 support vector"만이" 분류경계면 형성에 기여한다는 점을 고려하면 위와 같은 실험결과는 자연스러운 결과로 해석됩니다.

---

## Conclusion
지금까지 SVM의 이론적 배경, 개념, 그리고 이 이론적 내용을 증명하기 위한 실험 진행과정과 결과들을 살펴보았습니다. 다양한 개념을 살펴보고 이를 실제로 실험해보는 과정은 저에게 있어서는 SVM에 대한 이해와 머신러닝에 대한 이해가  과정이었는데, 이 tutorial을 보시는 분들에게도 그러했으면 좋겠습니다. 실제 실험의 진행 코드는 SVM_tutorial.ipynb에 있으며, 제가 참고한 자료와 실습을 진행한 데이터셋 heart.csv은 올려두었습니다. 감사합니다.


