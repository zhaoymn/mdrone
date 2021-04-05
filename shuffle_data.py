import numpy as np
sequences = list()
for i in range(0,100):
    sequences.append(i)
sequences = np.array(sequences)
np.random.shuffle(sequences)
train_sequences = sequences[0:80]
validation_sequences = sequences[80:90]
test_sequences = sequences[90:100]
np.savetxt('train_sequences.csv', train_sequences, fmt = '%d', delimiter = ',')
np.savetxt('validation_sequences.csv', validation_sequences, fmt = '%d', delimiter = ',')
np.savetxt('test_sequences.csv', test_sequences, fmt = '%d', delimiter = ',')
print(test_sequences)