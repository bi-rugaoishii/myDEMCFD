import numpy as np
import matplotlib.pyplot as plt

# parameters
r = 0.01
rho = 1000
theta = np.radians(60)
mu = 0.2
g = 9.81

t = np.linspace(0,2,1000)

# 判定条件
mu_crit = (2.0/7.0) * np.tan(theta)

if mu >= mu_crit:
    print("pure rolling")
    a = (5.0/7.0) * g * np.sin(theta)
else:
    print("slipping")
    a = g * (np.sin(theta) - mu * np.cos(theta))

# 速度
v = a * t

vx = v * np.cos(theta)
vy = -v * np.sin(theta)

plt.figure()
plt.plot(t,vx,label="vx")
plt.plot(t,vy,label="vy")

result = np.loadtxt("../results/pid0.txt",skiprows=1)
plt.plot(result[:,0], -result[:,5])
plt.plot(result[:,0], result[:,6])

plt.xlabel("t (s)")
plt.ylabel("velocity (m/s)")
plt.legend()
plt.show()
