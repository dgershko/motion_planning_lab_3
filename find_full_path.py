from environment import Environment
from kinematics import Transform, UR5e_PARAMS
from inverse_kinematics import get_inverse_kinematics_solutions
from building_blocks import Building_Blocks
from planners import RRT_STAR
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import config

def find_config_plan(conf_a, conf_b, cube_coords, step_size, rrt_iter):
    """
    Function for finding a plan between two configurations
    Must re-create the environment and building blocks objects
    """
    np.random.seed(os.getpid())
    ur_params = UR5e_PARAMS(inflation_factor=1)
    transform = Transform(ur_params)
    env = Environment(env_idx=3, cube_coords=cube_coords)
    bb = Building_Blocks(transform, ur_params, env, p_bias=0.2)
    planner = RRT_STAR(max_step_size=step_size, max_itr=rrt_iter, bb=bb)
    return planner.find_path(conf_a, conf_b, return_cost=False)


class PathOptimizer():
    def __init__(self, username):
        self.step_size = 1
        self.rrt_iter = 600
        self.optimizer_iter = 100
        self.optimize_existing_iter = 30
        self.num_cubes = 6
        self.pool = mp.Pool(16)
        self.ur_params = UR5e_PARAMS(inflation_factor=1)
        self.transform = Transform(self.ur_params)
        self.env = Environment(env_idx=3)
        self.bb = Building_Blocks(self.transform, self.ur_params, self.env)
        self.home = config.home
        self.cube_coords = config.cube_coords
        self.init_coords = self.cube_coords.copy()
        self.cube_approaches = config.cube_approaches
        self.cubes_actual = config.cubes_actual

        if username == "barakat":
            self.user_path = "paths/barakat"
            self.target_cube_coords = config.barakat_cubes
        elif username == "dgershko":
            self.user_path = "paths/dgershko"
            self.target_cube_coords = config.dgershko_cubes
        elif username == "roman":
            self.user_path = "paths/roman"
            self.target_cube_coords = config.roman_cubes

        self.existing_plans, self.existing_plans_quality = self.load_existing_plans(self.user_path)
            
        try:
            self.target_cube_configs = np.load(f"{self.user_path}/place_cube_configs.npy")
        except:
            self.target_cube_configs = [
                self.find_optimal_config(self.cube_approaches[cube_idx], self.target_cube_coords[cube_idx])
                for cube_idx in range(self.num_cubes)
            ]
            np.save(f"{self.user_path}/place_cube_configs.npy", self.target_cube_configs)

    def load_existing_plans(self, npy_path):
        existing_plans = []
        existing_plans_quality = []
        for cube_idx in range(len(self.cube_coords)):
            try:
                grip_plan = np.load(f"{npy_path}/grip_cube_{cube_idx+1}.npy")
                grip_plan_quality = self.get_plan_quality(grip_plan[:-1])
            except:
                grip_plan = np.array([])
                grip_plan_quality = np.inf
            try:
                place_plan = np.load(f"{npy_path}/place_cube_{cube_idx+1}.npy")
                place_plan_quality = self.get_plan_quality(place_plan)
            except:
                place_plan = np.array([])
                place_plan_quality = np.inf
            existing_plans.append([grip_plan, place_plan])
            existing_plans_quality.append([grip_plan_quality, place_plan_quality])
        return existing_plans, existing_plans_quality
    
    def get_plan_cost(self, plan):
        if len(plan) < 2:
            return np.inf
        cost = 0
        for i in range(len(plan) - 1):
            cost += self.bb.edge_cost(plan[i], plan[i+1])
        return cost

    def get_plan_quality(self, plan):
        if len(plan) < 2:
            return np.inf
        cost = self.get_plan_cost(plan)
        return cost # * np.log10(len(plan))

    def simplify_path(self, path: np.ndarray):
        # print(f"Path length before simplification: {len(path)}")
        # print(f"Path cost before simplification: {self.get_plan_cost(path)}")
        original_cost = self.get_plan_cost(path)
        cost_epsilon = 1.00
        new_cost = original_cost
        while len(path) > 2:
            # go over array and remove the point which results in the best new cost
            best_cost = np.inf
            best_index = -1
            for i in range(1, len(path) - 1):
                if not self.bb.local_planner(path[i-1], path[i+1]):
                    continue
                cost_change = self.bb.edge_cost(path[i-1], path[i+1]) - self.bb.edge_cost(path[i-1], path[i]) - self.bb.edge_cost(path[i], path[i+1])
                if cost_change < best_cost:
                    best_cost = cost_change
                    best_index = i
            if best_index == -1:
                # cannot remove any points without violating local planner
                break
            if new_cost > cost_epsilon * original_cost:
                # any change will increase the cost over the threshold
                break
            path = np.delete(path, best_index, axis=0)
            new_cost = self.get_plan_cost(path)

        # print(f"Path length after simplification: {len(path)}")
        # print(f"Path cost after simplification: {self.get_plan_cost(path)}")
        # print("=====================================")
        return path

    def find_optimal_config(self, near_config, goal_xyz):
        """
        Given a start config and goal coordinates (xyz), return the closest IK solution of the goal
        """
        goal_x = goal_xyz[0]
        goal_y = goal_xyz[1]
        solutions = get_inverse_kinematics_solutions(goal_x, goal_y)
        return min(solutions, key=lambda x: self.bb.edge_cost(near_config, x))
        # return min(solutions, key=lambda x: np.linalg.norm(near_config - x))

    def draw_target_cubes(self):
        print(self.target_cube_coords)
        sns.set_theme()
        sns.set_style("whitegrid")
        sns.scatterplot(x=[cube[0] for cube in self.target_cube_coords], y=[cube[1] for cube in self.target_cube_coords], s=100)
        # draw lines for xlim, ylim
        plt.axhline(y=-0.55, color='r', linestyle='--')
        plt.axhline(y=-0.19, color='r', linestyle='--')
        plt.axvline(x=-0.34, color='r', linestyle='--')
        plt.axvline(x=0.05, color='r', linestyle='--')
        plt.show()
    
    def optimize_plan(self):
        plans = []
        current_config = self.home
        for i in range(self.num_cubes):
            grip_plan = self.optimize_grip(current_config, i)
            current_config = self.cube_approaches[i]
            self.cube_coords.pop(i)
            place_plan = self.optimize_place(current_config, i)
            self.cube_coords.insert(i, self.target_cube_coords[i])
            current_config = self.target_cube_configs[i]
            plans.append([grip_plan, place_plan])
    
    def optimize_grip(self, start_config, cube_idx):
        num_iter = self.optimize_existing_iter if self.existing_plans_quality[cube_idx][0] < np.inf else self.optimizer_iter
        if num_iter == 0:
            return np.array([])
        print(f"Optimizing grip plan for cube {cube_idx+1} with {num_iter} iterations")
        starmap_args = [(self.cube_approaches[cube_idx], start_config, self.cube_coords, self.step_size, self.rrt_iter)] * num_iter
        # starmap_args = [(start_config, self.cube_approaches[cube_idx], self.cube_coords, self.step_size, self.rrt_iter)] * num_iter
        results = self.pool.starmap(find_config_plan, starmap_args)
        fix_grip_result = lambda res: np.vstack((self.simplify_path(res[::-1]), self.cubes_actual[cube_idx]))
        # fix_grip_result = lambda res: np.vstack((self.simplify_path(res), self.cubes_actual[cube_idx]))
        results = [fix_grip_result(res) for res in results if len(res) >= 2]
        if results == []:
            print(f"\033[91mNo valid grip plans found for cube {cube_idx+1}\033[0m")
            return np.array([])
        grip_plan = min(results, key=self.get_plan_quality)
        if self.get_plan_quality(grip_plan) < self.existing_plans_quality[cube_idx][0]:
            print(f"\033[92mSaving grip plan for cube {cube_idx+1}\033[0m")
            np.save(f"{self.user_path}/grip_cube_{cube_idx+1}", grip_plan)
        else:
            print(f"\033[91mExisting grip plan is better for cube {cube_idx+1}\033[0m")
        print(f"New best plan: cost: {self.get_plan_quality(grip_plan)}, length: {len(grip_plan)}")
        print(f"Existing plan: cost: {self.existing_plans_quality[cube_idx][0]}, length: {len(self.existing_plans[cube_idx][0])}")
        print(f"local planner cache: {self.bb.local_planner_cache_hits} hits, {self.bb.local_planner_cache_misses} misses, size = {len(self.bb.local_planner_cache)}")
        print("", flush=True)
        return grip_plan

    def optimize_place(self, start_config, cube_idx):
        num_iter = self.optimize_existing_iter if self.existing_plans_quality[cube_idx][0] < np.inf else self.optimizer_iter
        if num_iter == 0:
            return np.array([])
        print(f"Optimizing place plan for cube {cube_idx+1} with {num_iter} iterations")
        starmap_args = [(start_config, self.target_cube_configs[cube_idx], self.cube_coords, self.step_size, self.rrt_iter)] * num_iter
        results = self.pool.starmap(find_config_plan, starmap_args)
        fix_place_result = lambda res: self.simplify_path(np.vstack((self.cubes_actual[cube_idx], self.cube_approaches[cube_idx], res)))
        results = [fix_place_result(res) for res in results if len(res) >= 2]
        if results == []:
            print(f"\033[91mNo valid place plans found for cube {cube_idx+1}\033[0m")
            return np.array([])
        place_plan = min(results, key=self.get_plan_quality)
        if self.get_plan_quality(place_plan) < self.existing_plans_quality[cube_idx][1]:
            np.save(f"{self.user_path}/place_cube_{cube_idx+1}", place_plan)
            print(f"\033[92mSaving place plan for cube {cube_idx+1}\033[0m")
        else:
            print(f"\033[91mExisting place plan is better for cube {cube_idx+1}\033[0m")
        print(f"New best plan: cost: {self.get_plan_quality(place_plan)}, length: {len(place_plan)}")
        print(f"Existing plan: cost: {self.existing_plans_quality[cube_idx][1]}, length: {len(self.existing_plans[cube_idx][1])}")
        print(f"local planner cache: {self.bb.local_planner_cache_hits} hits, {self.bb.local_planner_cache_misses} misses, size = {len(self.bb.local_planner_cache)}")
        print("", flush=True)
        return place_plan

    def save_paths_to_gif(self):
        from gif_visualizer import VisualizeGif
        plans = []
        for cube_idx in range(self.num_cubes):
            grip_plan = np.load(f"{self.user_path}/grip_cube_{cube_idx+1}.npy")
            place_plan = np.load(f"{self.user_path}/place_cube_{cube_idx+1}.npy")
            plans.extend([grip_plan, place_plan])
        # generate list of current cube positions for each plan in plans
        cube_positions = []
        cubes = self.init_coords.copy()
        for cube_idx in range(6):
            cube_positions.append(cubes.copy())
            cubes.pop(cube_idx)
            cube_positions.append(cubes.copy())
            cubes.insert(cube_idx, self.target_cube_coords[cube_idx])

        gif_vis = VisualizeGif(self.ur_params, self.env, self.transform, self.bb)
        gif_vis.save_paths_to_gif(plans, cube_positions, f"{self.user_path}/cube_plan.gif")

if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(10000)
    path_optimizer = PathOptimizer("roman")
    path_optimizer.optimize_plan()
    path_optimizer.save_paths_to_gif()

