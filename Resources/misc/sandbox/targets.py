import numpy as np

batch_size = 5

# inputs are the states
batch_states = [[np.random.random((5,5)), np.random.random(3)] for _ in range(batch_size)]  # bx[40x40,3x1]

print(np.shape(batch_states))
print(batch_states[0])

batch_targets = [np.random.random((3,3)) for _ in range(batch_size)]  # bx3x3
print(np.shape(batch_targets))

batch_actions = [np.random.randint(3, size=3) for _ in range(batch_size)] # bx3
print(batch_actions)

# # get corresponding successor states for minibatch
# batch_suc_states = np.array([self.suc_states[i] for i in minibatch])  # bx[40x40,3x1]
batch_terminals = np.array([np.random.randint(2) for _ in range(batch_size)]).astype('bool')  # bx1
batch_rewards = np.array([np.random.random() for _ in range(batch_size)])  # bx1

print(batch_terminals)
print(batch_rewards)

Q_suc = [np.random.random((3,3)) for _ in range(batch_size)] # bx3x3

max_Q_suc = np.amax(Q_suc, axis=2) * np.repeat(np.expand_dims(np.invert(batch_terminals),axis=1), 3, axis=1) # bx3

print(np.shape(Q_suc), np.shape(max_Q_suc))
print(Q_suc)
print(max_Q_suc)

print("target:", batch_targets)

print(np.shape(batch_targets))

for i in range(batch_size):
    batch_targets[i][range(3), batch_actions[i]] = max_Q_suc[i] + batch_rewards[i]

    #print(batch_targets[range(3),range(3),batch_actions])

print("target':", batch_targets)

#batch_targets[range(batch_size), batch_actions] = max_Q_suc + batch_rewards
# # print("final targets:\n", batch_targets)
#
# self.model.train_on_batch(batch_states, batch_targets)