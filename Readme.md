# Optimization for Machine Learning project

Papers to get ideas on Forward Gradient:

https://paperswithcode.com/method/forward-gradient

Base paper code:
https://github.com/orobix/fwdgrad

Can Forward Gradient match Backprop (more complicated paper) code :
https://github.com/streethagore/ForwardLocalGradient

Scaling forward gradient with local losses (Google Paper) code:
https://github.com/google-research/google-research

## First meeting, TA's takes on the project: 
- Define clear hypothesis (for ex : fwd gradient is comparable to gd on large datasets)
- We don’t have to worry about the most recent state of art in fwd gradient, it's fine to start from the classic paper we first found
- Train on larger dataset (Cifar, Resnet)
- Find measure of time (flops) depending on hypothesis
- We can choose to train on any data. We have to clearly state our thinking and be transparent with results
- We can state new formulas (for example using momentum) without proving them. Writing the formula is enough
