# Batch_Gradient_Descent

# Overview
It demostrates the batch gradient descent algorithm of machine learning for linear regression problems.
Here I have used python but we can use any language(like, C,C++,Java,etc.) to perform the similar algorithm.

# Dependencies
- `pip3 install -U numpy`
- Use any toy dataset with simple one feature & label
- Simply run the main.py

# Explanation
This `Gradient Descent Algorithm` simply uses the one variable polynomial equation of straight line i.e.,
> Y = &theta;<sub>0</sub> + &theta;<sub>1</sub> * x

- &theta;<sub>1</sub> : slope of line
- &theta;<sub>0</sub> : Y intercept of the line (i.e., From which Y-coodinate the line will pass)
      
### Computing `Hypothesis` function:
**hypothesis()** - computes predicted/hypothysed value of corresponding value of input feature (x), given by
> h<sub>&theta;</sub>(x) = &theta;<sub>0</sub> + &theta;<sub>1</sub> * x<sup>(i)</sup>

### Computing `Cost` or `Error` function:
**cost()** - computes the totalError of the straight line which is calculated as `sum of squared error` by:
> cost(&theta;<sub>0</sub>, &theta;<sub>1</sub>) = 1/2M <sup>M</sup>&sum;<sub>(i=1)</sub> (h(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup>
corresponds to
> __cost(&theta;<sub>0</sub>, &theta;<sub>1</sub>) = 1/2M * <sup>M</sup>&sum;<sub>(i=1)</sub> ((&theta;<sub>0</sub> + &theta;<sub>1</sub> * x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup>__

### Computing `GradientDescent`:
#### To compute gradient descent we need to find the partial derivative of \theta_{0} and \theta_{1} respectively individually
- Formula to compute &theta;<sub>0</sub>'s partial derivative 
> &part; / &part; &theta;<sub>0</sub> = 1 / M <sup>M</sup>&sum;<sub>(i=1)</sub> ((&theta;<sub>0</sub> + &theta;<sub>1</sub> * x<sup>(i)</sup>) - y<sup>(i)</sup>)

- Formula to compute &theta;<sub>1</sub>'s partial derivative 
> &part; / &part; &theta;<sub>1</sub> = 1 / M <sup>M</sup>&sum;<sub>(i=1)</sub> ((&theta;<sub>0</sub> + &theta;<sub>1</sub> * x<sup>(i)</sup>) - y<sup>(i)</sup>) * x<sup>(i)</sup>

<h5>while convergence:<br>
  <pre>     </pre>&theta;<sub>0</sub> = &theta;<sub>0</sub> - &alpha; * 1/M * <sup>M</sup>&sum;<sub>(i=1)</sub> ((&theta;<sub>0</sub> + &theta;<sub>1</sub> * x<sup>(i)</sup>) - y<sup>(i)</sup>)<br>
  <pre>     </pre>&theta;<sub>1</sub> = &theta;<sub>1</sub> - &alpha; * 1/M * <sup>M</sup>&sum;<sub>(i=1)</sub> ((&theta;<sub>0</sub> + &theta;<sub>1</sub> * x<sup>(i)</sup>) - y<sup>(i)</sup>) * x<sup>(i)</sup><br>
  return [&theta;<sub>0</sub>, &theta;<sub>1</sub>]</h5>
      
#### epcohs is simply the number of iterations we want to make over the whole dataset repetatively.
#### &alpha; is simply the learning rate of the gradient descent function.
