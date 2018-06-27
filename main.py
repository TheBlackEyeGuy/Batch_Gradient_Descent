import numpy as np

def hypothesis(b,m,x):
	predicted_y = (b + (m*x))
	return predicted_y

def cost(b,m,points):
	totalError = 0
	M = (float(len(points)))
	for i in range(len(points)):
		x = float(points[i,0])
		y = float(points[i,1])
		totalError += (1/(2*M)) * ((hypothesis(b,m,x) - y)**2)
	return totalError

def gradient_descent(b,m,points,alpha,epochs):
	partial_b=partial_m=0
	new_b = b
	new_m = m
	for convergence in range(epochs):
		for i in range(len(points)):
			x = float(points[i,0])
			y = float(points[i,1])
			partial_b += (1/float(len(points))) * (hypothesis(new_b,new_m,x) - y)
			partial_m += (1/float(len(points))) * (hypothesis(new_b,new_m,x) - y) * x
		new_b = (b - (alpha*partial_b))
		new_m = (m - (alpha*partial_m))
	return [new_b, new_m]

def main():
	b=m=0
	alpha = 0.0008
	epochs = 1000
	points = np.genfromtxt('data.csv', delimiter=',')
	[new_b, new_m] = gradient_descent(b, m, np.array(points), alpha, epochs)
	x = float(input("Enter X value to predict : "))
	y = hypothesis(new_b,new_m,x)
	print('Predicted price is : {}'.format(y))

	print("Cost function's weights are b:{}, m:{}".format(new_b, new_m))
	print("Total cost is : {}".format(cost(new_b,new_m,np.asarray(points))))
	
if __name__ == '__main__':
	main()
	
