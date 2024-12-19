import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import numpy as np

def create_wheels(axis1X, axis2X):
    w = []

    for i in [axis1X, axis2X]:
        wheel_front = Circle((0, 2), radius=0.8, fill=False, edgecolor='black')
        wheel_back = Circle((0 + 0.05, 2), radius=0.8, fill=False, edgecolor='black')
        grf.add_patch(wheel_front)
        grf.add_patch(wheel_back)
        w += [wheel_front, wheel_back]

    return w

def create_axis(x, y):
    mesh = [(x - 1, y), (x + 1, y), (x, y - 2)]
    axis = Polygon(mesh, closed=True, fill=False, edgecolor='green')
    grf.add_patch(axis)
    return [mesh, axis]

m1, m2, m3, m4 = 10, 3, 0.1, 1
g = 9.81
alpha = np.radians(15)
F0, omega = 3, 0.5
speed_multiplier = 3.5

theta = np.arctan2(-3, 4)

t_eval = np.linspace(0, 20, 500)
A = 2.0
omega = 0.5
x_vals = A * np.sin(omega * t_eval) 
s_vals =A * np.sin(omega * t_eval)


fgr, grf = plt.subplots()

[wheel1, wheel2, wheel3, wheel4] = create_wheels(0, 5)

plane_mesh = [(-100, 1.2), (100, 1.2)]
plane = Polygon(plane_mesh, closed=False, fill=False, edgecolor='black')
grf.add_patch(plane)

[axis1_mesh, axis1] = create_axis(0, 4)
[axis2_mesh, axis2] = create_axis(5, 4)

pillar_mesh = [(2.6, 8.4), (2.4, 8), (2.8, 8)]
pillar = Polygon(pillar_mesh, closed=True, fill=False, edgecolor='yellow')
grf.add_patch(pillar)

body_mesh = [(-3, 4), (-3, 8), (3, 8), (7, 5), (7, 4)]
body = Polygon(body_mesh, closed=True, fill=False, edgecolor='green')
grf.add_patch(body)

little_circle = Circle((2.6, 8.6), radius=0.4, fill=False, edgecolor='blue')
big_circle = Circle((-1, 8.8), radius=0.8, fill=False, edgecolor='red')
grf.add_patch(little_circle)
grf.add_patch(big_circle)

cylinder = Polygon([(0, 0), (0.6, 0), (0.6, 0.45), (0, 0.45)], closed=True, fill=False, edgecolor='#E1592F')
grf.add_patch(cylinder)

[line1] = grf.plot([0, 0], [0, 0], color='black')
[line2] = grf.plot([0, 0], [0, 0], color='black')

direction_vector = np.array([-4, 3]) / 5

def rotate_cylinder(base_center, width, height, theta):
    local_mesh = np.array([
        [-width / 2, 0],
        [width / 2, 0],
        [width / 2, height],
        [-width / 2, height]
    ])
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    rotated_mesh = local_mesh @ rotation_matrix.T + base_center
    return rotated_mesh

def move_cart(dx, ds):
    wheel1.center = (dx, 2)
    wheel2.center = (dx + 5, 2)
    wheel3.center = (dx + 0.1, 2)
    wheel4.center = (dx + 0.1 + 5, 2)
    body_xy = np.array(body_mesh) + [dx, 0]
    body.set_xy(body_xy)

    axis1_xy = np.array(axis1_mesh) + [dx, 0]
    axis1.set_xy(axis1_xy)

    axis2_xy = np.array(axis2_mesh) + [dx, 0]
    axis2.set_xy(axis2_xy)

    pillar_xy = np.array(pillar_mesh) + [dx, 0]
    pillar.set_xy(pillar_xy)
    little_circle.center = (2.6 + dx, 8.4)

def update_rope(cylinder_mesh):
    cylinder_center = cylinder_mesh.mean(axis=0)

    line1.set_xdata([little_circle.center[0], big_circle.center[0]])
    line1.set_ydata([little_circle.center[1] + 0.4, big_circle.center[1]])

    line2.set_xdata([little_circle.center[0], cylinder_center[0]])
    line2.set_ydata([little_circle.center[1] + 0.4, cylinder_center[1]])

def anim(i):
    dx = x_vals[i]
    ds = s_vals[i]

    move_cart(dx, ds)

    horizontal_offset =  -s_vals[i]
    big_circle.center = (-0.8 + dx + horizontal_offset, 8.8)

    rect_base_center = np.array([3 + dx, 8]) + (ds - 3) * direction_vector
    cylinder_mesh_transformed = rotate_cylinder(rect_base_center, 1.2, 0.6, theta)
    cylinder.set_xy(cylinder_mesh_transformed)

    update_rope(cylinder_mesh_transformed)

    return wheel1, wheel2, body, axis1, axis2, pillar, little_circle, big_circle, cylinder

grf.set_xlim(-5, 10)
grf.set_ylim(0, 10)
grf.set_aspect('equal')
ani = FuncAnimation(fgr, anim, frames=500, interval=50)

plt.show()