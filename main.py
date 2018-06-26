import numpy as np

def hypothesis(b,m,x):
	predicted_y = (b + (m*x))
	return predicted_y

def cost(b,m,points):
	totalError = 0
	M = float(len(points))
	for i in range(len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError += (hypothesis(b,m,x) - y)**2
	return totalError / (2*M)

def gradient_descent(b,m,points,alpha):
	partial_b=partial_m=0
	for i in range(len(points)):
		x = points[i,0]
		y = points[i,1]
		partial_b += (hypothesis(b,m,x) - y)
		partial_m += (hypothesis(b,m,x) - y) * x
	
	partial_b /= float(len(points))
	partial_m /= float(len(points))
	
	new_b = (b - (alpha*partial_b))
	new_m = (m - (alpha*partial_m))
	
	return [new_b, new_m]

def linear_regression(b=0,m=0,points=None,alpha=0.00005,epochs=1000):
	new_b = b
	new_m = m
	for i in range(epochs):
		new_b, new_m = gradient_descent(b,m,points,alpha)
	return [new_b, new_m]

def main():
	points = np.genfromtxt('data.csv', delimiter=',')
	new_b, new_m = linear_regression(b=0, m=0, points=np.array(points), alpha=0.0004, epochs=1000)
	x=int(input("Enter X value to predict : "))
	y = hypothesis(new_b,new_m,x)
	print('Predicted price is : {}'.format(y))

	print("Cost function's weights are b:{}, m:{}".format(new_b, new_m))
	print("Total cost is : {}".format(cost(new_b,new_m,np.array(points))))
	
if __name__ == '__main__':
	main()
	
	
