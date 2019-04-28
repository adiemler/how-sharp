import matplotlib.pyplot as plt
import numpy as np

alphas = np.linspace(-1, 2, 25)

data = np.zeros((25, 4))
with open('alpha.csv', 'r') as f:
    i = 0
    for line in f:
        if i == 0:
            pass  # trash header
        else:
            terms = line.split(',')
            terms[3] = terms[3][:-1]
            data[i-1, :] = terms #[float(t) for t in terms]
        i += 1

fig, axis1 = plt.subplots()
#fig.suptitle('R1 Parametric Plot')
axis2 = axis1.twinx()
axis1.plot(alphas, data[:, 0], 'b-')
axis1.plot(alphas, data[:, 2], 'b--')

axis2.plot(alphas, data[:, 1]*100., 'r-')
axis2.plot(alphas, data[:, 3]*100., 'r--')

axis1.set_xlabel('Alpha')
axis1.set_ylabel('Loss', color='b')
axis2.set_ylabel('Accuracy', color='r')
axis1.legend(('Train', 'Test'), loc=0)

axis1.grid(b=True, which='both')
plt.savefig('r1-alpha.png')