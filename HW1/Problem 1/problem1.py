import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import collections 
plt.style.use('ggplot')


Data = pd.read_csv('https://math189r.github.io/hw/data/online_news_popularity/' 'online_news_popularity.csv', sep=', ', engine='python')


Data['cohort'] = 'train'
Data.iloc[int(0.666*len(Data)):int(0.833*len(Data)),-1] = 'validation' 
Data.iloc[int(0.833*len(Data)):,-1] = 'test'

X_train = Data[Data.cohort == 'train'][[col for col in Data.columns if col not in ['url', 'shares', 'cohort']] ]
y_train = np.log(Data[Data.cohort == 'train'].shares).reshape(-1,1)

X_val = Data[Data.cohort == 'validation'][[col for col in Data.columns if col not in ['url', 'shares', 'cohort']] ]
y_val = np.log(Data[Data.cohort == 'validation'].shares).reshape(-1,1)

X_test = Data[Data.cohort == 'test'][[col for col in Data.columns if col not in ['url', 'shares', 'cohort']] ]
y_test = np.log(Data[Data.cohort == 'test'].shares).reshape(-1,1)

X_train = np.hstack((np.ones_like(y_train), X_train))
X_val = np.hstack((np.ones_like(y_val), X_val))
X_test = np.hstack((np.ones_like(y_test), X_test))

X_train_M = np.matrix(X_train)
X_train_T = np.transpose(X_train_M)

X_val_M = np.matrix(X_val)
X_val_T = np.transpose(X_val_M)

X_test_M = np.matrix(X_test)
X_test_T = np.transpose(X_test_M)

#################
#				#
#    Part C     #
#				#
################# 


def linreg(X, XT, y, reg=0.0):
	eye = np.eye(X.shape[1])
	eye[0,0] = 0.
	return np.linalg.solve(
		XT * X + reg * eye,
		XT * y )


lambda_L = []
thetaNorm_L = []
RMSE_L = []

for index in range(150):
	# randomly generate lambda from 0 to 150
	lambda_i = random.uniform(0.0, 150.0)
	lambda_L += [lambda_i]

	# Calculate the theta using linreg fxn above and then calculate the norm
	theta = linreg(X_train_M, X_train_T, y_train, reg=lambda_i)
	theta_norm = np.linalg.norm(theta)
	thetaNorm_L += [theta_norm]

	# Compute the y estimation using the theta calculated for validation set
	y = X_val_M * theta

	# Compute the RMSE of y prediction and actual y_val
	sum_Error = 0
	for j in range(len(y)):
		diff = y_val.item(j) - y.item(j)
		diff_square = diff ** 2
		sum_Error += diff_square

	RMSE_i = (sum_Error / len(y)) ** (0.5)
	RMSE_L += [RMSE_i]



# Get the minimum RMSE for validation set for all 150 different lambda
# the lambda with min RMSE is the optimal lambda
RMSE_min = min(RMSE_L)
Optimal_index = RMSE_L.index(RMSE_min)
lambda_optimal = lambda_L[Optimal_index]
theta_optimal = linreg(X_train_M, X_train_T, y_train, reg=lambda_optimal)

# Use the optimal lambda obtained to predict y of testset
y_estimate = X_test_M * theta_optimal

# Computer the RMSE of test set using optimal lambda
sum_Error = 0
for i in range(len(y_estimate)):
	diff = y_test.item(i) - y_estimate.item(i)
	diff_square = diff ** 2
	sum_Error += diff_square

RMSE_test = (sum_Error / len(y_estimate)) ** (0.5)


plot_dict = {}
for index in range(len(lambda_L)):
	plot_dict[lambda_L[index]] = RMSE_L[index]

oplot_dict = collections.OrderedDict(sorted(plot_dict.items()))


new_lambda = oplot_dict.keys()
new_RMSE = oplot_dict.values()
#new_RMSE = []
#for d_index in range(len(oplot_dict)):
#	new_lambda += [oplot_dict[d_index]]


plt.plot(new_lambda, new_RMSE)
plt.ylabel('RMSE')
plt.xlabel('lambda')
plt.show()

print "Lambda Optimal is: {}".format(lambda_optimal)
print "RMSE is : {}".format(RMSE_min)
print "RMSE_test is {}:".format(RMSE_test)



#plt.plot(lambda_L, thetaNorm_L)
#plt.ylabel('Norm of Theta')
#plt.xlabel('Lambda')
#plt.title('Norm of Theta vs. Lambda')
#plt.show()

#plt.plot(lambda_L, RMSE_L)
#plt.ylabel('RMSE of Validation Set')
#plt.xlabel('lambda')
#plt.title('RMSE of Validation Set vs. Lambda')
#plt.show()


#################
#				#
#    Part D     #
#				#
################# 

X_train_no_b = X_train_M[:,1:]
X_train_no_b_T = np.transpose(X_train_no_b)
n = len(X_train_no_b)

X_train_mod = X_train_no_b_T * (np.eye(n) - (1.0 / n) * np.ones((n,n)))
m = len(X_train_mod)

theta_opt2 = np.linalg.solve (
		X_train_mod * X_train_no_b + lambda_optimal * np.eye(m),
		X_train_mod * y_train )

intercept = (y_train - X_train_no_b * theta_opt2).sum() / n

distance_b = abs(theta_optimal.item(0) - intercept)
distance_theta = np.linalg.norm(theta_opt2 - theta_optimal[1:])

print "Distance between the intercept calculated in part c and d is : {}".format(distance_b)
print "Distance between the optimal theta calculated in part c and d is : {}".format(distance_theta)


#################
#				#
#    Part E     #
#				#
#################  

shape = (X_train_no_b.shape[1],1)
precision = 1e-6
max_iteration = 500
lr_theta = 2.5e-13
lr_intercept = 0.2

X_val_no_b = X_val_M[:,1:]
X_val_no_b_T = np.transpose(X_val_no_b)
X_train_mod = X_train_no_b_T * X_train_no_b
Lambda_M = lambda_optimal * np.eye(shape[0])
y_train_sum = y_train.sum()

theta = np.zeros(shape)
intercept = 0.

grad_theta = np.ones_like(theta)
grad_intercept = np.ones_like(intercept)

grad_theta_norm = np.linalg.norm(grad_theta)
grad_intercept_norm = abs(grad_intercept)


RMSE_train_GD = []
RMSE_GD = []
while grad_theta_norm > precision and grad_intercept_norm > precision and len(RMSE_train_GD) < max_iteration:
	
	y_train_estimate = X_train_no_b * theta + intercept

	sum_Error_t = 0
	for j in range(len(y_train_estimate)):
		diff = y_train.item(j) - y_train_estimate.item(j)
		diff_square = diff ** 2
		sum_Error_t += diff_square

	RMSE_train_i = (sum_Error_t / len(y_train_estimate)) ** (0.5)
	RMSE_train_GD += [RMSE_train_i]

	y_val_estimate = X_val_no_b * theta + intercept

	sum_Error_v = 0
	for j in range(len(y_val_estimate)):
		diff = y_val.item(j) - y_val_estimate.item(j)
		diff_square = diff ** 2
		sum_Error_v += diff_square

	RMSE_i = (sum_Error_v / len(y_val_estimate)) ** (0.5)
	RMSE_GD += [RMSE_i]

	grad_theta = ((X_train_mod + Lambda_M) * theta + X_train_no_b_T * (intercept - y_train)) / n
	grad_intercept = ((X_train_no_b * theta).sum() - y_train_sum + intercept * n) / n

	theta = theta - lr_theta * grad_theta
	intercept = intercept - lr_intercept * grad_intercept
 
distance_b = abs(theta_optimal.item(0) - intercept)
distance_theta = np.linalg.norm(theta - theta_optimal[1:])


print "Distance between the intercept calculated from gradient descent and direct computation closed form solution  : {}".format(distance_b)
print "Distance between the theta optimal calculated from gradient descent and direct computation closed form solution  : {}".format(distance_theta)



plt.plot(range(500), RMSE_GD, color='red')
plt.plot(range(500), RMSE_train_GD, color='blue')
plt.show()


