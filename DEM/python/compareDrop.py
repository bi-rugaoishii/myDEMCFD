import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# parameters
g = -9.81
r = 0.01
floor = -1.0
e = 0.3          # 反発係数
dt = 0.001
tmax = 2.0

# initial conditions
y = 0.5
v = 0.0

t_list = []
y_list = []

t = 0.0

while t < tmax:

    # 保存
    t_list.append(t)
    y_list.append(y)

    # 運動
    v += g * dt
    y += v * dt

    # 衝突判定
    if y - r < floor:
        y = floor + r
        v = -e * v

    t += dt

result = np.loadtxt("../results/pid0.txt",skiprows=1)

plt.plot(t_list, y_list)
plt.plot(result[:,0], result[:,3])
plt.xlabel("t (s)")
plt.ylabel("y (m)")
plt.title("Bouncing particle")
plt.show()
