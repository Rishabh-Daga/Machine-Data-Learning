import numpy
import pickle
import matplotlib.pyplot as plt
from operator import add,sub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pickle.load(open('../Q1_data/data.pkl', 'rb'))

train, test = train_test_split(data, test_size = 0.1)
training_parts = numpy.array(numpy.array_split(train, 10))

X_test = test[:,0].reshape(-1,1)
y_test = test[:,1].reshape(-1,1)

test_data_size = X_test.size
total_bias = []
total_variance = []

Input = int(input("Enter the number of maximum degree:"))
while ( Input > 10 or Input < 1 ):
	Input = int(input("Please enter the input within the permitted constraints: "))

for degree in range(Input):
		
	list_of_outputs = []
	output_total = numpy.zeros(y_test.shape)
	variance_array = numpy.zeros(y_test.shape)

	p = PolynomialFeatures(degree + 1)

	for i in range(len(training_parts)):
		
		X_train = training_parts[i][:,0].reshape(-1,1)
		y_train = training_parts[i][:,1].reshape(-1,1)

		X_train_polynomial = p.fit_transform(X_train)        
		X_test_polynomial = p.fit_transform(X_test)

		regr = LinearRegression()
		regr.fit(X_train_polynomial,y_train)

		output = regr.predict(X_test_polynomial).reshape(-1,1)
		output_total = list(map(add, output_total, output))
		list_of_outputs.append(output)

	arr_of_outputs = numpy.asarray(list_of_outputs)

	expectation_output = (numpy.array(output_total)/len(training_parts)).reshape(-1,1)

	bias_array = numpy.square(list(map(sub, y_test, expectation_output))).reshape(-1,1)
	bias_square = numpy.mean(bias_array)
	total_bias.append(bias_square)
	bias = bias_square**(0.5)

	for w in range(test_data_size):
		for k in range(len(training_parts)):
			variance_array[w] += (pow((arr_of_outputs[k][w][0] - expectation_output[w]),2))
	
	variance = numpy.mean(variance_array) 
	variance /= len(training_parts)
	total_variance.append(variance) 

	string = ("Degree = {} , BIAS^2 = {} , BIAS = {} , VARIANCE = {}")
	print(string.format(degree + 1, bias_square,bias, variance))

x_axis = numpy.arange(1, Input + 1)
plt.plot(x_axis, total_bias, label = "Bias^2", marker = '*', color = 'green')
plt.plot(x_axis, total_variance, label = "Variance", marker = '*', color = 'red')
plt.xlabel('x-axis')
plt.ylabel('Bias-Variance')
plt.legend()
plt.show()
