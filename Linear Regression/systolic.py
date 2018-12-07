#http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html
import numpy as numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
df=pd.read_excel('/Users/bhavyababuta/Downloads/mlr02.xls')
X=df.as_matrix()

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

plt.scatter(X[:,1],X[:,0])
plt.show()

plt.scatter(X[:,2],X[:,0])
plt.show()

X=df['X1','X2']
Y=df['X3']
