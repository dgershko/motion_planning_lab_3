from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks

from gif_visualizer import VisualizeGif

from cProfile import Profile
import numpy as np

target_cube_coords = [
    [-0.205, -0.475, 0],
    [-0.115, -0.475, 0],
    [-0.115, -0.415, 0],
    [-0.205, -0.355, 0],
    [-0.115, -0.355, 0],
    [-0.115, -0.265, 0],
]
# start = np.array([ 1.11036637, -1.41822527,  2.10266132,  0.88636028,  1.57079633, -0.46042996])
# end = np.array([ 1.3665928,  -1.07686815,  2.11010307, -1.52890842, -0.22514747,  0.4834562 ])
home = np.deg2rad([0, -90, 0, -90, 0, 0])
env = Environment(env_idx=3)
params = UR5e_PARAMS()
transform = Transform(params)
bb = Building_Blocks(transform, params, env)
g_viz = VisualizeGif(params, env, transform, bb)
g_viz.draw(home, np.array(target_cube_coords))
# planner = RRT_STAR(1.0, 500, bb)
# profiler = Profile()

# profiler.enable()
# planner.find_path(start, end)
# profiler.disable()

# profiler.print_stats('cumtime')