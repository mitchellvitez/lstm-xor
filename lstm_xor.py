"""
LSTM that solves XOR - Mitchell Vitez 2020

Input: binary sequences of length SEQ_LEN
Output: 0 or 1, whether the number of 1s in the sequence is odd

See also the problem statement at https://openai.com/blog/requests-for-research-2
"""

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Input, LSTM, Activation
from tensorflow.keras.models import Sequential
import numpy as np
import random

SEQ_LEN = 50
COUNT = 100000

bin_pair = lambda x: [x, not(x)]
training = np.array([[bin_pair(random.choice([0, 1])) for _ in range(SEQ_LEN)] for _ in range(COUNT)])
target = np.array([[bin_pair(x) for x in np.cumsum(example[:,0]) % 2] for example in training])

print('shape check:', training.shape, '=', target.shape)

model = Sequential()
model.add(Input(shape=(SEQ_LEN, 2), dtype='float32'))
model.add(LSTM(1, return_sequences=True))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training, target, epochs=10, batch_size=128)
model.summary()

predictions = model.predict(training)
i = random.randint(0, COUNT)
chance = predictions[i,-1,0]
print('randomly selected sequence:', training[i,:,0])
print('prediction:', int(chance > 0.5))
print('confidence: {:0.2f}%'.format((chance if chance > 0.5 else 1 - chance) * 100))
print('actual:', np.sum(training[i,:,0]) % 2)
