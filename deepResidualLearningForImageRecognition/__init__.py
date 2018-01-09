import numpy as np
from matplotlib import pyplot as plt

t = np.arange(0, 4,1)  # x轴上的点，0到2之间以0.01为间隔
s = [4,3,5,7]
t1 = np.arange(0,4,1)
s2 = [4,4,6,8]
plt.plot(t, s)  # 画图
plt.plot(t1,s2)
plt.xlabel('time (s)')  # x轴标签
plt.ylabel('voltage (mV)')  # y轴标签
plt.title('About as simple as it gets, folks')  # 图的标签
plt.grid(True)  # 产生网格
plt.savefig("test.png")  # 保存图像
plt.show()  # 显示图像