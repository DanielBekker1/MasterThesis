from keras.models import Sequential
from keras.layers import Dense
import h5py

def build_actor_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    return model

# Example usage
state_shape = (7,)  # Example state shape
action_size = 10  # Example action size
actor_model = build_actor_model(state_shape, action_size)

# Save the model to an HDF5 file
actor_model.save('actor_model.h5')