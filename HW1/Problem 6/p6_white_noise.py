import numpy
import matplotlib.pyplot as plt

# calculate the noise 
noise = numpy.random.normal(0,1,size=100)

# theta: m (slope) and b (intercept)
m = 62.0 / 35.0
b = 18.0 / 35.0

# x coordinate of the fit line
sample_x = numpy.linspace(0, 20, 100)
sample_y = []

# y coordinate of the fit line
for i in range(100):
	sample_y.append(m * sample_x[i] + b + noise[i])

# Matrix X
A= numpy.matrix([[x,1] for x in sample_x])
# Matrix X^T
At = A.transpose()

# Calculate (X * X^T)^(-1) * Y^T
result = numpy.linalg.inv(At * A) * At * numpy.matrix(sample_y).transpose()
# Get the new m
new_m = result.item(0)
# Get the new b
new_b = result.item(1)


plt.plot(sample_x, sample_y, 'ro')
plt.plot(sample_x, sample_x * m + b)
plt.plot(sample_x, sample_x * new_m + new_b)

plt.show()

