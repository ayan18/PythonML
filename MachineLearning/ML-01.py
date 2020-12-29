#%%
# Load CSV Using Python Standard Library
# Load CSV from URL using NumPy
from pandas import read_csv
from pandas import set_option
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import numpy
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names=names)
print(data.shape)
# print(data)

########################################################
# Chapter 5: Understand Your Data With Descriptive Statistics
########################################################
#peek = data.head(20)
#print(peek)
types = data.dtypes
print(types)

set_option('display.width', 100)
set_option('precision', 1)
description = data.describe()
print(description)

class_counts = data.groupby('class').size()
print(class_counts)

set_option('display.width', 100)
set_option('precision', 1)
correlations = data.corr(method = 'pearson')
print('Correlation %:')
print(correlations*100)

skew = data.skew()
print("Skew:", skew)

########################################################
# Chapter 6: Understand Your Data With Visualization
########################################################
print('Histogram')
data.hist(density=True)
pyplot.show()

#Plot density
#data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
#pyplot.show()

print('Box and Whisker Plots')
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

print('Plot correlation matrix')
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

print('Scatter plot matrix')
scatter_matrix(data)
pyplot.show()

########################################################
# Chapter 7: Prepare Your Data For Machine Learning
########################################################
array = data.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

print('MinMaxScaler:')
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(X[0:5, :])
print(rescaledX[0:5,:])
print(DataFrame.from_records(rescaledX).describe())

print('StandardScaler:')
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])
print(DataFrame.from_records(rescaledX).describe())
