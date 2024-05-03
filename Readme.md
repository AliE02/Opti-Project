# Optimization for Machine Learning project

Papers to get ideas on Forward Gradient:

https://paperswithcode.com/method/forward-gradient

**Base paper code :** https://github.com/orobix/fwdgrad

**Can Forward Gradient match Backprop** : They use local gradients from auxiliary losses as Gradient Guess directions for the computation of forward gradient with respect to a Global Target. The study confirms that  Gradient Guesses obtained from Local Guesses exceeded the performance of Random Guesses. In the case of Gradient Guesses using supervised Local Losses, a consistent positive alignment between the Gradient Targets and Guesses improves the Forward Gradient estimation.  They conclude that stimating consistently the Global Target should be the main focus of Forward Gradient algorithms. Evaluation done on ResNet-18 architecture divided into 8 and 16 blocks on the standard object recognition tasks CIFAR10, ImageNet32, CIFAR100, MNIST.

code :
https://github.com/streethagore/ForwardLocalGradient

**Scaling forward gradient with local losses (Google Paper)** :  They reduce the variance of the forward gradient estimator by applying perturbations to activations rather than weights. They also improve the scalability of forward gradient by introducing a large number of local greedy loss functions, each of which involves only a small number of learnable parameters. This approach matches backprop on MNIST and CIFAR-10 and significantly outperforms previously proposed forward gd algorithms on ImageNet.   

code : https://github.com/google-research/google-research/tree/master/local_forward_gradient

## First meeting, TA's takes on the project: 
- Define clear hypothesis (for ex : fwd gradient is comparable to gd on large datasets)
- We don’t have to worry about the most recent state of art in fwd gradient, it's fine to start from the classic paper we first found
- Train on larger dataset (Cifar using Resnet)
- Find measure of time (flops) depending on hypothesis
- We can choose to train on any data. We have to clearly state our thinking and be transparent with results
- We can state new formulas (for example using momentum) without proving them. Writing the formula is enough

## Second meeting:
We came up with 3 hypothesis :
- Can Forward gd match Backpropagation accross different domains?
- Assessment of the performance of Forward gd across usual ML architectures
- Forward vs Backward : What is the difference in performance accross different domains, datasets and architectures?

We need to run fwgd on different datasets and tasks :
- Andrea : Nlp
- Ali : Audio and speech
- Yassine : Computer vision

We should use a profiler to count flops (or any other technique we can find to account for time in the best possible way).

We should try to use the base paper code architecture (hydra, SGD with exponential lr decay).

## Third meeting
Tuesday 7th May



