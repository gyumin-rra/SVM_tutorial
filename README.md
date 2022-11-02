# SVM tutorial
a simple tutorial for support vector machine

이 repository는 support vector machine에 대해 아무것도 모르는 분들을 위해 작성되었습니다. 그래서 우선 support vector machine에 대한 기본적인 아이디어와 증명을 다루고 이를 기반으로 support vector machine을 다른 모듈의 구현체없이 직접 구현해보는 순서로 구성하였습니다. 그리고 이 repository의 이론적인 토대는 첨부드린 논문과 고려대학교 강필성 교수님의 [유튜브 강의](https://www.youtube.com/watch?v=gzbafL28vA0&list=PLetSlH8YjIfWMdw9AuLR5ybkVvGcoG2EW&index=8)를 참고하였음을 밝힙니다. 

## 목차
1. [Theoretical Background](#theoretical-background)
2. [Concepts of Support Vector Machine(SVM)](#concepts-of-support-vector-machinesvm)
3. [SVM Implementation](#svm-implementation)
4. [Conclusion](#conclusion)

---

## Theoretical Background
support vector machine(SVM)이 활발하게 쓰인 시점은 1998년 이후부터 2010년대 초반까지였습니다. 이후 딥러닝 계열의 알고리즘들이 SVM을 뛰어넘으면서 점차 SVM에 대한 열기가 사그라들었지만, 어쨌든 딥러닝 계열의 알고리즘들은 그 쓸모를 인정받기 위해 반드시 비교대상으로 넣어야 했던 알고리즘이 바로 SVM이었습니다. SVM이 활발하게 쓰일 당시 많은 머신러닝 알고리즘들은 shatter, VC dimension, structural/empirical risk minimization 등의 여러 이론적 배경을 기반하고 있는데요, 본격적인 SVM 소개에 앞서 이를 간단하게나마 먼저 짚고 넘어가보도록 하겠습니다. 

### Shatter
본론부터 말하자면 어떤 데이터셋 $S$가 있을 때, 어떤 함수 집합 $H$가 각 데이터 객체가 가질 수 있는 모든 이진분류 경우의 수(dichotomies)에 대해 이를 모두 성공적으로 분류할 수 있다면, $S$는 $H$에 의해 shatter된다($S$ is shattered by $H$)고 합니다. 이게 무슨 소리인지 한번 예시를 들어보겠습니다. 

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

---

## Concepts of Support Vector Machine(SVM)

---

## SVM Implementation


우선 실제 구현 및 실험에 앞서 필요한 모듈 등의 버젼은 아래와 같습니다.
| env_name   | version |
|------------|---------|
| python     | 3.8.3   |
| numpy      | 1.19.2  |
| matplotlib | 3.5.2   |
| pandas     | 1.4.3   |
| sklearn    | 1.1.1   |

구현해주어야 할 함수와 그를 구현한 결과는 아래와 같습니다.(numpy를 사용하였습니다.)
1. euclidean distance matrix 반환: n by d 데이터셋의 n by n 거리 행렬을 반환하는 함수.
```python
def make_dist_matrix(X):# n by d의 np.ndarray dataset 가정
    sum_sqr_X = np.sum(np.square(X), axis = 1)# 1 by n의 객체 element 제곱의 합 matrix
    dist_matrix = np.add(np.add(sum_sqr_X, -2*np.dot(X, X.T)).T, sum_sqr_X)# 1 diag(X X^T)^T -2 * X X^T + 1^T diag(X X^T)
    return dist_matrix
```
2. $p_{j|i}$ matrix 반환: distance matrix와 객체 별 sigma vector를 input으로 받아 i행 j열 원소에 $p_{j|i}$를 가지는 matrix 반환하는 함수
```python
def make_p_j_cond_i_mat(dist_matrix, sigma_vec):
    sqrd_sigma_vec = 2. * np.square(sigma_vec.reshape((-1, 1)))
    tmp_matrix = -dist_matrix / sqrd_sigma_vec
    exp_matrix = np.exp(tmp_matrix)
    np.fill_diagonal(exp_matrix, 0.) # p_i|i == 0
    exp_matrix = exp_matrix + 1e-10 # avoiding division by 0 
  
    return exp_matrix / exp_matrix.sum(axis=1).reshape([-1, 1])
```
3. $p_{ij}$ matrix 반환: $p_{j|i}$ matrix를 받아 $p_{ij}$ matrix를 반환하는 함수 
```python
def make_p_ij_mat(p_j_cond_i_mat):
    return (p_j_cond_i_mat + p_j_cond_i_mat.T) / (2. * p_j_cond_i_mat.shape[0])
```
4. $q_{ij}$ matrix 반환: 축소된 공간에서의 데이터셋 n by d'를 받아 $q_{ij}$ matrix와 그래디언트 계산을 위한 $1+|y_i-y_j|^2$행렬을 반환하는 함수
```python
def make_q_ij_mat(Y):
    dist_matrix = make_dist_matrix(Y)
    invrs_dist_mat = np.power(1. + dist_matrix, -1)
    np.fill_diagonal(invrs_dist_mat, 0.) # q_ii == 0
    
    return invrs_dist_mat / np.sum(invrs_dist_mat), invrs_dist_mat# for gradient
```
5. binary search: 이진탐색 구현
```python
def binary_search(func, target, lower_bound=1e-20, upper_bound=1000., tolerance=1e-10, max_iter=5000):
    for i in range(max_iter):
        guess = (lower_bound+upper_bound)/2
        guess = func(guess) # function will be perplexity calculator from sigma
        if np.abs(guess - target) <= tolerance:
            break
        
        if guess > target:
            upper_bound = guess
        else:
            lower_bound = guess
    
    return guess
```
6. perplexity vector 반환: 거리행렬과 sigma vector를 input으로 하여 각 객체들의 현재 perplexity를 원소로하는 벡터를 반환하는 함수
```python
def make_perp_vec(dist_matrix, sigma_vec):
    p_j_cond_i_mat = make_p_j_cond_i_mat(dist_matrix, sigma_vec)
    entropy = -np.sum(p_j_cond_i_mat * np.log2(p_j_cond_i_mat), 1)# j에 대해 모두 더함.
    perp_vec = 2 ** entropy# 1 by n perplexity 벡터 
    return perp_vec
```
7. sigma vercotr 반환: 5, 6을 이용하여 설정한 perplexity를 만족하는 sigma를 찾는 함수
```python
def make_sigma_vec(dist_matrix, target_perplexity, make_perp_vec):
    sigma_vec = [] 
    for i in range(dist_matrix.shape[0]):
        func = lambda sigma: \
            make_perp_vec(dist_matrix[i:i+1, :], np.array(sigma)) # 객체 i에 대한 perplexity 계산
        
        correct_sigma = binary_search(func, target_perplexity)
        
        sigma_vec.append(correct_sigma)
    # 1 by n sigma_vec 반환
    return np.array(sigma_vec)
```
8. gradient matrix 반환: solution을 update하기 위한 gradient를 update하는 함수
```python
def make_grad_matrix(p_ij_mat, q_ij_mat, Y, invrs_dist_mat):
    pq_diff_mat = p_ij_mat - q_ij_mat
    pq_expanded = np.expand_dims(pq_diff_mat, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    distances_expanded = np.expand_dims(invrs_dist_mat, 2)

    grad_matrix = 4. * (pq_expanded * y_diffs * distances_expanded).sum(1)
    return grad_matrix
```
9. optimization: 초기화 후 momentum을 이용한 최적화를 하는 함수
```python
def optimization(X, p_ij_mat, max_iter, learning_rate, momentum, target_dim, seed):
    # initialization
    Y = np.random.default_rng(seed=1).normal(0.0, scale = 0.0001, size = [X.shape[0], target_dim])
    Y_t = Y.copy()# momentum
    Y_t_1 = Y.copy()# momentum

    # gradient descent
    for i in range(max_iter):
        # q_ij 구하기
        q_ij_mat, invrs_dist_mat = make_q_ij_mat(Y)
        # gradient 구하기
        grad_matrix = make_grad_matrix(p_ij_mat, q_ij_mat, Y, invrs_dist_mat)

        # solution update
        Y = Y - learning_rate*grad_matrix + momentum*(Y_t - Y_t_1)
        # momentum update
        Y_t_1 = Y_t.copy()
        Y_t = Y.copy()
            
    return Y
```
10. t-SNE 함수: 원 논문에서의 TSNE 수도코드를 구현한 함수. time 모듈을 활용하여 시간을 측정하였습니다.
```python
def raw_TSNE(X, target_dim, target_perplexity, max_iter, learning_rate, momentum, seed):
    import time
    st = time.time()
    dist_matrix = make_dist_matrix(X)
    print('made dist matrix, ' + str(time.time()-st))
    sigma_vec = make_sigma_vec(dist_matrix, target_perplexity, make_perp_vec)
    print('made sigma vector, ' + str(time.time()-st))
    p_ij_mat = make_p_ij_mat(make_p_j_cond_i_mat(dist_matrix, sigma_vec))
    print('made p_ij_mat, ' + str(time.time()-st))
    Y = optimization(X, p_ij_mat, max_iter, learning_rate, momentum, target_dim, seed)
    print('done, ' + str(time.time()-st))
    
    return Y
```

이를 이용하여 [MNIST](http://yann.lecun.com/exdb/mnist/)를 기반으로 실제로 tSNE를 진행한 결과는 아래와 같습니다.(pandas(v1.19.2)와 matplotlib(v1.19.2)을 활용하였습니다.)
```python
import gzip
import numpy as np
from raw_tsne import raw_TSNE
with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    y_train = np.frombuffer(f.read(), np.uint8, offset=8)

X = x_train[0:1000]
label = y_train[0:1000]

tsne_data = raw_TSNE(X, 2, 40, 1000, 200, 0.5, 1013)

import pandas as pd
import matplotlib.pyplot as plt
tsne_data = pd.DataFrame(tsne_data, columns=['z1', 'z2'])
plt.figure(figsize=(20,20))
plt.title('MNIST, raw_t-sne')
plt.scatter(tsne_data.z1, tsne_data.z2, c=label, alpha=0.7, cmap=plt.cm.tab10)
```
![image](https://user-images.githubusercontent.com/112034941/195624702-1b39c3c0-e505-4303-8dc5-59abc4868272.png)
![image](https://user-images.githubusercontent.com/112034941/195624783-1bca7146-7ece-49dd-8115-3965dd28a5ad.png)


결과를 보면, manifold 학습이 잘 되지 않았고 모든 MNIST trainset을 학습한 것이 아니라 일부만 학습한 것임에도 오랜 시간(약 247초)이 걸림을 알 수 있습니다. 같은 결과를 sklearn의 TSNE를 통해 구현한 결과는 아래와 같습니다. 
```python
import gzip
import numpy as np
from raw_tsne import raw_TSNE
with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    y_train = np.frombuffer(f.read(), np.uint8, offset=8)

X = x_train[0:1000]
label = y_train[0:1000]

from sklearn.manifold import TSNE
import time
import warnings 
warnings.filterwarnings("ignore") # warning 무시
tsne = TSNE(n_components=2, random_state = 1013)
st = time.time()
tsne_data = tsne.fit_transform(X)
et = time.time()
tsne_data = pd.DataFrame(tsne_data, columns=['z1', 'z2'])
print(et-st)
plt.figure(figsize=(20,20))
plt.title('MNIST, sklearn_TSNE')
plt.scatter(tsne_data.z1, tsne_data.z2, c=label, alpha=0.7, cmap=plt.cm.tab10)
```
![image](https://user-images.githubusercontent.com/112034941/195560135-053c61b8-72a9-4c6a-9666-fcd1e4865c5a.png)

결과를 보면 실제 결과와 시간과 성능 면에서 심한 차이가 있음을 알 수 있습니다. 하이퍼 파라미터의 차이도 그 이유 중 하나겠지만, 그보다 중요한 것은 현재 구현된 t-SNE 코드는 실제 t-SNE논문의 'simple version of t-distributed Stochastic Neighbor Embedding'의 구현체라는 점입니다. 편의를 위해 이 t-SNE 구현체를 raw t-SNE라고 하면, 실제로 요즈음 쓰이는 t-SNE에는 raw t-SNE에 몇 가지 내용이 추가됩니다. 

1. Ealry Exaggeration: 조금 더 빠른 해의 수렴을 위해 $p_{ij}$ 행렬에 특정 수를 곱하고, 일정 iteration을 넘어가면 다시 원래 행렬로 변환하여 gradient를 계산합니다.
2. [Barnes-hut SNE](https://arxiv.org/pdf/1301.3342.pdf): metric trees 및 barnes-hut algorithm을 $p_{ij}$ 행렬을 근사하고 그래디언트도 근사하여 계산복잡도를 줄입니다. 본래 계산복잡도는 $O(N^2)$이지만, 이 방법론에서의 해당 과정의 계산복잡도는 $O(NlogN)$이라고 합니다.
3. Numerical Stability: $p_{ij}$ 행렬과 $q_{ij}$ 행렬에서 너무 값이 작아지는 것을 방지하고자 $1e-12$(다른 값도 가능합니다.) 보다 낮은 값이 들어있는 경우 이를 $1e-12$로 대체합니다. 
4. Using PCA: PCA를 통해 먼저 차원을 축소한 후 이를 가지고 t-SNE를 사용합니다. 
5. 기타: 파이썬이 아니라 실제로는 c기반으로 계산한 후 이를 파이썬으로 wrapping하여 계산 속도 자체롤 높입니다.

위와 같은 요소가 최근 사용되는 t-SNE 구현체와 이 repository에서의 구현체와의 차이점입니다. 여기에 적절한 하이퍼파라미터의 설정 또한 성능을 높이기 위해 필요한 점입니다. 

---

## Conclusion
지금까지 t-SNE의 개념, 구현, 기존 모듈과의 비교 등을 진행해보았습니다. t-SNE의 이론적 배경 분석부터 가장 기본적인 형태의 코드 구현까지 이어지는 과정이 저에게는 쉽지 않았던 만큼 t-SNE에 대한 이해가 조금이나마 깊어질 수 있었는데, 여기까지 읽으신 분들에게도 그랬으면 좋겠습니다. 실험 결과의 경우에는 t-SNE_tutorial.ipynb에 있으며, 혹여나 이 markdown에 구현된 raw t-SNE를 진행하고 싶으신 분들을 위해 raw_tsne.py 파일과 실습을 진행한 데이터셋은 올려두었습니다. 감사합니다.


