import numpy as np
import h5py
import Cense.NeuralNetworkFactory.nnFactory as factory


def save_array():
    states = []
    actions = []
    rewards = []
    suc_states = []
    terminals = []

    experience = []

    for i in range(50):
        state = np.random.rand(50, 50)
        action = np.random.rand(6, 1)
        reward = np.random.uniform(-10, 10)
        suc_state = np.random.rand(50, 50)
        terminal = np.random.randint(2)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        suc_states.append(suc_state)
        terminals.append(terminal)

    f = h5py.File('dummy_data.h5', 'w')

    f.create_dataset('states', data=states)
    f.create_dataset('actions', data=actions)
    f.create_dataset('rewards', data=rewards)
    f.create_dataset('suc_states', data=suc_states)
    f.create_dataset('terminals', data=terminals)
    f.close()


def load_array():
    f = h5py.File('dummy_data.h5', 'r')

    data = f['states'][:]
    print(data.shape)

    f.close()


def save_model():
    model = factory.model_simple_conv((50, 50), 6)

    with open('model.json', 'w') as file:
        file.write(model.to_json())

    model.save_weights("weights.h5")


if __name__ == "__main__":
    # save_array()

    # load_array()

    save_model()
