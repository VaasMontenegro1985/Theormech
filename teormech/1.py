import math
import sympy as s
import matplotlib.pyplot as plot
import numpy as np
from matplotlib.animation import FuncAnimation

t = s.Symbol('t')

r = 1 + s.sin(5 * t)
phi = t + 0.3 * s.sin(30 * t)

x = r * s.cos(phi)
y = r * s.sin(phi)

x_v = s.diff(x)
y_v = s.diff(y)

x_acc = s.diff(x_v)
y_acc = s.diff(y_v)

v = s.sqrt(x_v ** 2 + y_v ** 2)
acc = s.sqrt(x_acc ** 2 + y_acc ** 2)
acc_t = s.diff(v)
acc_n = s.sqrt(acc ** 2 - acc_t ** 2)
RCrv = (v ** 2) / acc_n

step = 2000

T = np.linspace(0, 10, step)

X = np.zeros_like(T)
Y = np.zeros_like(T)

X_v = np.zeros_like(T)
Y_v = np.zeros_like(T)

X_acc = np.zeros_like(T)
Y_acc = np.zeros_like(T)

X_rCrv = np.zeros_like(T)
Y_rCrv = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = s.Subs(x, t, T[i])
    Y[i] = s.Subs(y, t, T[i])

    X_v[i] = s.Subs(x_v, t, T[i])
    Y_v[i] = s.Subs(y_v, t, T[i])

    X_acc[i] = s.Subs(x_acc, t, T[i])
    Y_acc[i] = s.Subs(y_acc, t, T[i])

    v_angle = math.atan2(Y_v[i], X_v[i])
    acc_angle = math.atan2(Y_acc[i], X_acc[i])
    RCrv_angle = v_angle - math.pi / 2 if v_angle - acc_angle > 0 else v_angle + math.pi / 2

    X_rCrv[i] = RCrv.subs(t, T[i]) * math.cos(RCrv_angle)
    Y_rCrv[i] = RCrv.subs(t, T[i]) * math.sin(RCrv_angle)

def rotate(x, y, angle):
    Rot_x = x * np.cos(angle) - y * np.sin(angle)
    Rot_y = x * np.sin(angle) + y * np.cos(angle)
    return Rot_x, Rot_y

def make_arrow_head(VecX, VecY, X, Y):
    arrow_x = np.array([-0.3, 0, -0.3])
    arrow_y = np.array([0.2, 0, -0.2])

    phi = math.atan2(VecY, VecX)

    RotX, RotY = rotate(arrow_x, arrow_y, phi)

    arrow_x = RotX + X + VecX
    arrow_y = RotY + Y + VecY

    return arrow_x, arrow_y

fgr = plot.figure()

grf = fgr.add_subplot(1, 1, 1)
grf.axis('equal')
grf.set(xlim=[-10, 10], ylim=[-10, 10])
grf.plot(X, Y, 'grey')

Pnt = grf.plot(X[0], Y[0])
circle = plot.Circle((X[0], Y[0]), 0.1, color='#2DBA20')

grf.add_patch(circle)

X_RArrow, Y_RArrow = make_arrow_head(X[0], Y[0], 0, 0)
RArrow = grf.plot(X_RArrow, Y_RArrow, 'blue')[0]
RVector = grf.plot([0, X[0]], [0, Y[0]], 'blue')[0]

X_VArrow, Y_VArrow = make_arrow_head(X_v[0], Y_v[0], X[0], Y[0])
VArrow = grf.plot(X_VArrow, Y_VArrow, 'r')[0]
VVector = grf.plot([X[0], X[0] + X_v[0]], [Y[0], Y[0] + Y_v[0]], 'r')[0]

X_AArrow, Y_AArrow = make_arrow_head(X_acc[0], Y_acc[0], X[0], Y[0])
AArrow = grf.plot(X_AArrow, Y_AArrow, 'g')[0]
AVector = grf.plot([X[0], X[0] + X_acc[0]], [Y[0], Y[0] + Y_acc[0]], 'g')[0]

X_RCArrow, Y_RCArrow = make_arrow_head(X_rCrv[0], Y_rCrv[0], X[0], Y[0])
RCArrow = grf.plot(X_RCArrow, Y_RCArrow, 'y')[0]
RCVector = grf.plot([X[0], X[0] + X_rCrv[0]], [Y[0], Y[0] + Y_rCrv[0]], 'y')[0]

def anim(i):
    circle.set(center=(X[i], Y[i]))

    RVector.set_data([0, X[i]], [0, Y[i]])
    RArrow.set_data(make_arrow_head(X[i], Y[i], 0, 0))

    VVector.set_data([X[i], X[i] + X_v[i]], [Y[i], Y[i] + Y_v[i]])
    VArrow.set_data(make_arrow_head(X_v[i], Y_v[i], X[i], Y[i]))

    AVector.set_data([X[i], X[i] + X_acc[i]], [Y[i], Y[i] + Y_acc[i]])
    AArrow.set_data(make_arrow_head(X_acc[i], Y_acc[i], X[i], Y[i]))
    
    RCVector.set_data([X[i], X[i] + X_rCrv[i]], [Y[i], Y[i] + Y_rCrv[i]])
    RCArrow.set_data(make_arrow_head(X_rCrv[i], Y_rCrv[i], X[i], Y[i]))

    return

an = FuncAnimation(fgr, anim, frames=step, interval=50)

plot.show()