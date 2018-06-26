# Batch_Gradient_Descent

# Overview
It demostrates the batch gradient descent algorithm of machine learning for linear regression problems.
Here I have used python but we can use any language(like, C,C++,Java,etc.) to perform the similar algorithm.

# Dependencies
- `pip3 install -U numpy`
- Use any toy dataset with simple one feature & label
- Simply run the main.py

# Explanation
This `Gradient Descent Algorithm` simply uses the one variable polynomial equation of straight line i.e., `Y = mx + b`
Here, m - slope of line
      b - Y intercept of the line (i.e., From which Y-coodinate the line will pass)
      
### Computing `Hypothesis` function:
hypothesis() - computes predicted/hypothysed value of corresponding value of input feature (x), given by
> h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub> * x<sup>(i)</sup>

### Computing `Cost` or `Error` function:
cost() - computes the totalError of the straight line which is calculated as `sum of squared error` by:
> cost(&theta;<sub>0</sub>, &theta;<sub>1</sub>) = 1/2M \sum<sub>i=1</sub><sup>M</sup> (h(x<sup>(i)</sup> - y<sup>(i)</sup>)
corresponds to > cost(\theta_{0}, \theta_{1}) = /frac{1}{2M} * \sum_{i=1}^{M} ((\theta_{0} + \theta_{1}*x^(i)) - y^(i))

### Computing `GradientDescent`:
#### To compute gradient descent we need to find the partial derivative of \theta_{0} and \theta_{1} respectively individually
-Formula to compute \theta_{0}'s partial derivative `\frac{d}{d\theta_{0}} = \frac{1}{M} \sum_{i=1}^{M} ((\theta_{0} + \theta_{1}*x^(i)) - y^(i))`
-Formula to compute \theta_{1}'s partial derivative `\frac{d}{d\theta_{1}} = \frac{1}{M} \sum_{i=1}^{M} ((\theta_{0} + \theta_{1}*x^(i)) - y^(i)) * x^(i)`
```
while convergence:
  \theta_{0} = \theta_{0} - \alpha * \sum_{i=1}^{M}((\theta_{0} + \theta{1}*x^(i)) - y^(i))
  \theta_{1} = \theta_{1} - \alpha * \sum_{i=1}^{M}((\theta_{1} + \theta{1}*x^(i)) - y^(i)) * x^(i)
  return [\theta_{0}, \theta_{1}]
```
#### epcohs is simply the number of iterations we want to make over the whole dataset repetatively.
#### \alpha is simply the learning rate of the gradient descent function.
