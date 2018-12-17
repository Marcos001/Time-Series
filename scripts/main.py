

from CNN.pre_data import split_sequence
from CNN.models import simple_cnn_ts
from numpy import array

# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps
n_steps = 3

# split into samples
X, y = split_sequence(raw_seq, n_steps)

# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# get input model
input_model = (n_steps, n_features)

# define model
model = simple_cnn_ts(input=input_model)

# fit model
print('-'*30)
print('training')
model.fit(X, y, epochs=1000, verbose=1)
print('-'*30)

print('-'*30)
print('test')
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
y_pred = model.predict(x_input, verbose=0)
print(x_input, ' = ', y_pred)
