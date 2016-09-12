import numpy
import matplotlib.pyplot as plt

# theta: m (slope) and b (intercept)
m = 62.0 / 35.0
b = 18.0 / 35.0

# x coordinate of the fit line 
sample_x = numpy.linspace(0, 8, 8)
sample_y = []

# y coordinate of the fit line 
for i in range(8):
	sample_y.append(m * sample_x[i] + b)


#Plot the date sets
plt.plot([0,2,3,4],[1,3,6,8],'ro')
#Plot the fit line 
plt.plot(sample_x, sample_x * m + b)
#Show the plot 
plt.show()
