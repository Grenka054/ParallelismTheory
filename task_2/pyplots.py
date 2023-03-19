import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors
import numpy as np
# cat_par = [f"{i}*{i}" for i in [128, 256, 512, 1024]]

# g1 = [0.5, 0.67, 100, 0] 
# g2 = [1.1, 6.5, 45, 340]

# width = 0.3

# x = np.arange(len(cat_par))

# fig, ax = plt.subplots()

# rects1 = ax.bar(x - width/2, g1, width, label='Onecore')
# rects2 = ax.bar(x + width/2, g2, width, label='Multicore')

# ax.set_title('Onecore and multicore time, s')
# ax.set_xticks(x)
# ax.set_xticklabels(cat_par)
# ax.set_axisbelow(True)
# ax.legend()
# fig.set_figheight(7)
# plt.grid(axis = 'y')
# fig.savefig('cpu.png')



cat_par = [f"{i+1}" for i in range(5)]

g1 = [0.95, 0.73, 0.7, 0.5, 0.25]

width = 0.3

x = np.arange(len(cat_par))

fig, ax = plt.subplots()

rects1 = ax.bar(x, g1, width, label='GPU time, s')

ax.set_title('Этапы оптимизации')
ax.set_xticks(x)
ax.set_xticklabels(cat_par)
ax.set_axisbelow(True)
ax.legend()
fig.set_figheight(3)
plt.grid(axis = 'y')
fig.savefig('opt.png')





# cat_par = [f"{i}*{i}" for i in [128, 256, 512, 1024]]

# g1 = [0.5, 0.67, 100, 0] 
# g2 = [1.1, 6.5, 45, 340]
# g3 = [0.5, 0.8, 3.1, 34]
# width = 0.3

# x = np.arange(len(cat_par))

# fig, ax = plt.subplots()

# rects1 = ax.bar(x - width, g1, width, label='Onecore')
# rects2 = ax.bar(x, g2, width, label='Multicore')
# rects3 = ax.bar(x + width, g3, width, label='GPU')
# ax.set_title('Onecore and multicore time, s')
# ax.set_xticks(x)
# ax.set_xticklabels(cat_par)
# ax.set_axisbelow(True)
# ax.legend()
# fig.set_figheight(7)
# plt.grid(axis = 'y')
# fig.savefig('gpu.png')