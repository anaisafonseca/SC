import numpy as np
import matplotlib.pyplot as plt

# Para uso de entrada quando apropriado (ver instruções)
t = np.arange(0,70,0.1)
u = np.zeros(t.shape)
u[t>=0] = 0
u[t>=5] = 0.5
u[t>=10] = 0.8
u[t>=12] = 0
u[t>=25] = 1
u[t>=60] = 0.5
u[t>=50] = 0.8
u[t>=60] = 0

# Apenas para demonstração
plt.plot(t, u)
plt.show()
