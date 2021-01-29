from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

# clean up memory
#K.clear_session()

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

#print(train_targets)

#give deviation
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

#need use mutiple times, so build a function
def build_model():
    model = models.Sequential()
    #activation: tanh
    #model.add(layers.Dense(64, activation='tanh',input_shape=(train_data.shape[1],)))
    #model.add(layers.Dense(64, activation='tanh'))

    #32 hidden units
    #model.add(layers.Dense(32, activation='relu',input_shape=(train_data.shape[1],)))
    #model.add(layers.Dense(32, activation='relu'))

    #128 hidden units
    #model.add(layers.Dense(128, activation='relu',input_shape=(train_data.shape[1],)))
    #model.add(layers.Dense(128, activation='relu'))

    #stock test
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))

    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 4 #partitions
num_val_samples = len(train_data) // k #diviation
#num_epochs = 100 #give epochs number
num_epochs = 500
all_mae_histories = []
#all_scores = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    #first train model
    #model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=1, verbose=0)
    # Evaluate the model on the validation data
    #val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    #all_scores.append(val_mae)

    #500 epochs
    #history = model.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=0)
    #print(history.history.keys())
    #mae_history = history.history['val_mae']
    #all_mae_histories.append(mae_history)

    # final model
    #model = build_model()
    model.fit(train_data, train_targets,
            epochs=80, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

#print(all_scores)
#print(np.mean(all_scores))
#print()

#compute average of the pre epoch mae scores
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#show plot
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()




print(test_mae_score)