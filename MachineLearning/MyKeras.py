#%%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
data = np.random.random((1000, 100))
lables = np.random.randint(2, size = (1000, 1))
model = Sequential()
model.add(Dense(32, 
                activation = 'relu',
                input_dim=100))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimiszer='rmsprop',
                loss = 'binary_crossntropy',
                metrics=['accuracy'])
model.fit(data, labels, epoch=10, batch_size=32)
predictions = model.predict(data)