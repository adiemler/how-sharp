import matplotlib.pyplot as plt

import numpy as np

epochs = [i for i in range(1, 51)]

data = np.zeros((50, 3))
with open('batchsizes.csv', 'r') as f:
    i = 0
    for line in f:
        if i == 0:
            pass  # trash header
        else:
            terms = line.split(',')
            data[i-1, :] = terms #[float(t) for t in terms]
        i += 1

#plt.figure(figsize=(6.4, 4))
plt.ylabel('Batch Size')
plt.xlabel('Epoch')
lq = plt.plot(epochs, data[:, 0], 'b-')
ss = plt.plot(epochs, data[:, 1], 'g-')
cq = plt.plot(epochs, data[:, 2], 'r-')
plt.legend(('LQ', 'SS', 'CQ'), 'lower right')

plt.savefig('bs.png')
plt.show()