import numpy as np
import matplotlib.pyplot as plt
#
hist = np.array([2,3,11,7,5,3,2,1,1,2])   # 10
idx = np.array([0,2,4,6,8,10,12,14,16,18,20]) # 11


x_med = np.array([])

for i in range(len(hist)) :
    x_med = np.append(x_med, (idx[i]+idx[i+1])/2.0)

x_med = x_med.tolist()
n_groups = len(x_med)

#
hist = hist.tolist()
plt.bar(x_med, hist, align='center')
plt.axis([min(x_med),max(x_med),0,max(hist)])
plt.show()
plt.isinteractive()