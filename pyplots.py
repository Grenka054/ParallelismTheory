import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import matplotlib.colors as mcolors
import numpy as np

cpu_one = [0.1, 1.8, 25, 0] 
cpu_multi = [0.5, 3.5, 20, 145]
gpu = [0.3, 0.5, 1.5, 16.7]
gpu_opt = [0.95, 0.73, 0.7, 0.5, 0.25, 0.23]

cat_par = [f"{i}*{i}" for i in [128, 256, 512, 1024]]
width = 0.3

x = np.arange(len(cat_par))

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, cpu_one, width, label='Onecore')
rects2 = ax.bar(x + width/2, cpu_multi, width, label='Multicore')

ax.set_title('Onecore and multicore time, s')
ax.set_xticks(x)
ax.set_xticklabels(cat_par)
ax.set_axisbelow(True)
ax.legend()
fig.set_figheight(7)
plt.grid(axis = 'y')
fig.savefig('cpu.png')



cat_par = [f"{i+1}" for i in range(6)]



width = 0.3

x = np.arange(len(cat_par))

fig, ax = plt.subplots()

rects1 = ax.bar(x, gpu_opt, width, label='GPU time, s')

ax.set_title('Этапы оптимизации')
ax.set_xticks(x)
ax.set_xticklabels(cat_par)
ax.set_axisbelow(True)
ax.legend()
fig.set_figheight(3)
plt.grid(axis = 'y')
fig.savefig('opt.png')





cat_par = [f"{i}*{i}" for i in [128, 256, 512, 1024]]

width = 0.3

x = np.arange(len(cat_par))

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, cpu_one, width, label='Onecore')
rects2 = ax.bar(x, cpu_multi, width, label='Multicore')
rects3 = ax.bar(x + width, gpu, width, label='GPU')
ax.set_title('Onecore and multicore time, s')
ax.set_xticks(x)
ax.set_xticklabels(cat_par)
ax.set_axisbelow(True)
ax.legend()
fig.set_figheight(7)
plt.grid(axis = 'y')
fig.savefig('gpu.png')