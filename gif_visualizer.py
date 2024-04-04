import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import imageio

from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from building_blocks import Building_Blocks

class VisualizeGif():
    def __init__(self, ur_params, env, transform, bb):
        self.ur_params = ur_params # type: UR5e_PARAMS
        self.env = env # type: Environment
        self.transform = transform # type: Transform
        self.bb = bb # type: Building_Blocks
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(0, 2)

    def draw_obstacles(self, cube_obstacles = []):
        u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
        for sphere in self.env.obstacles:
            x = np.cos(u) * np.sin(v) * self.env.radius + sphere[0]
            y = np.sin(u) * np.sin(v) * self.env.radius + sphere[1]
            z = np.cos(v) * self.env.radius + sphere[2]
            self.ax.plot_surface(x, y, z, color='red',alpha=0.5)
        for cube in cube_obstacles:
            # cube size is 4*4*4 cm
            # cube is defined by x,z,y of its center
            # approximate the cube as a sphere cause it's easier
            x = np.cos(u) * np.sin(v) * self.env.radius + cube[0]
            y = np.sin(u) * np.sin(v) * self.env.radius + cube[1]
            z = np.cos(v) * self.env.radius + cube[2]
            self.ax.plot_surface(x, y, z, color='blue', alpha=0.5)

    def draw_robot(self, config):
        global_sphere_coords = self.transform.conf2sphere_coords(config)
        u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
        # the global sphere coords is a dict, where link names are keys, and values are a list of spheres
        for link, spheres in global_sphere_coords.items():
            sphere_radius = self.ur_params.min_sphere_radius[link]
            link_color = self.ur_params.ur_links_color[link]
            for sphere in spheres:
                x = np.cos(u) * np.sin(v) * sphere_radius + sphere[0]
                y = np.sin(u) * np.sin(v) * sphere_radius + sphere[1]
                z = np.cos(v) * sphere_radius + sphere[2]
                self.ax.plot_surface(x, y, z, color=link_color, alpha=0.5)
        # return end effector position
        return global_sphere_coords['wrist_3_link'][-1]

    def draw_ee(self, ee_list):
        if len(ee_list) > 1:
            for i in range(len(ee_list) - 1):
                self.ax.plot(
                    [ee_list[i][0], ee_list[i+1][0]], 
                    [ee_list[i][1], ee_list[i+1][1]], 
                    [ee_list[i][2], ee_list[i+1][2]], color='red')

    def draw(self, config, cubes):
        self.draw_obstacles(cubes)
        self.draw_robot(config)
        plt.show()

    def interpolate_path(self, path):
        interpolated_path = []
        step_size = 0.2
        for i in range(len(path) - 1):
            config_a = path[i]
            config_b = path[i + 1]
            num_steps = int(np.linalg.norm(config_b - config_a) / step_size)
            points = np.linspace(config_a, config_b, num=num_steps, endpoint=False)
            interpolated_path.extend(points)
        interpolated_path.append(path[-1])
        return interpolated_path

    def save_paths_to_gif(self, path_list: list[list[list[float]]], cube_list: list[list[list[float]]], filename: str = None):
        """
        Inputs:
        path_list: a list of paths for the robot to take (list[list[configuration]])
        cube_list: a list of cube positions, for each of the paths (list[list[cube position]])
        each path needs a matching cube position list for where the cubes are during that path
        """
        num_paths = len(path_list)
        num_current_path = 1
        images = []
        end_effector_points = []
        for path, cubes in zip(path_list, cube_list):
            print(f'Drawing path {num_current_path} / {num_paths}')
            num_current_path += 1
            i_path = self.interpolate_path(path)
            for config in i_path:
                self.draw_obstacles(cubes)
                ee = self.draw_robot(config)
                end_effector_points.append(ee)
                self.draw_ee(end_effector_points)
                self.fig.canvas.draw()
                image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)
                self.ax.clear()
                self.ax.set_xlim3d(-1, 1)
                self.ax.set_ylim3d(-1, 1)
                self.ax.set_zlim3d(0, 2)
        if filename:
            imageio.mimsave(filename, images, fps=10)
        else:
            imageio.mimsave('animation.gif', images, fps=10)
        plt.close()