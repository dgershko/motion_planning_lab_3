from environment import Environment
from kinematics import Transform, UR5e_PARAMS
from inverse_kinematics import get_inverse_kinematics_solutions
from building_blocks import Building_Blocks
from planners import RRT_STAR
import numpy as np
from visualizer import Visualize_UR
from gif_visualizer import VisualizeGif
import seaborn as sns
import matplotlib.pyplot as plt
import os

ur_params = UR5e_PARAMS(inflation_factor=1)
transform = Transform(ur_params)


def get_plan_quality(plan):
    if plan is None:
        return np.inf
    env = Environment(env_idx=3)
    bb = Building_Blocks(transform, ur_params, env, p_bias=0.2)
    cost = 0
    for i in range(len(plan) - 1):
        cost += bb.edge_cost(plan[i], plan[i+1])
    return cost * len(plan)

def find_config_plan(conf_a, conf_b, cube_coords):
    np.random.seed(os.getpid())
    env = None
    env = Environment(env_idx=3, cube_coords=cube_coords)
    bb = None
    bb = Building_Blocks(transform, ur_params, env, p_bias=0.2)
    planner = RRT_STAR(max_step_size=1, max_itr=700, bb=bb)
    # for i in range(6):
    #     path, cost = planner.find_path(conf_a, conf_b, return_cost=True)
    #     if path is not None:
    #         break
    return planner.find_path(conf_a, conf_b, return_cost=True)


def find_optimal_config(near_config, goal_xyz, obstacle_cubes):
    env = Environment(env_idx=3, cube_coords=obstacle_cubes)
    bb = Building_Blocks(transform, ur_params, env)
    goal_x = goal_xyz[0]
    goal_y = goal_xyz[1]
    solutions = get_inverse_kinematics_solutions(goal_x, goal_y)
    return min(solutions, key=lambda x: np.linalg.norm(near_config - x))


def visualize_full_plan(plans, init_cubes: list, target_cubes: list):
    cubes = init_cubes
    for cube, plan in enumerate(plans):
        grip_plan = plan[0]
        place_plan = plan[1]
        env = Environment(env_idx=3, cube_coords=cubes)
        bb = Building_Blocks(transform, ur_params, env)
        visualizer = Visualize_UR(ur_params, env, transform, bb)
        visualizer.show_path(grip_plan, sleep_time=0)

        cubes.pop(cube)
        cubes.insert(cube, target_cubes[cube])
        env = Environment(env_idx=3, cube_coords=cubes)
        bb = Building_Blocks(transform, ur_params, env)
        visualizer = Visualize_UR(ur_params, env, transform, bb)
        visualizer.show_path(place_plan, sleep_time=0)


if __name__ == "__main__":
    home = np.deg2rad([0, -90, 0, -90, 0, 0])
    cube1_coords = [-0.10959248574268822, -0.6417732149769166, 0.1390226933317033]
    cube2_coords = [0.08539928976845282, -0.8370930220946053, 0.13813472317717034]
    cube3_coords = [-0.008445229140271685, -0.7365370847309188, 0.00955541284784159]
    cube4_coords = [0.23647185443765273, -0.769747539513382, 0.03971366463235271]
    cube5_coords = [0.26353072323141574, -0.4629969534200313, 0.2651034131371637]
    cube6_coords = [0.26940059242703984, -0.4730222745248458, 0.021688493137064376]
    cube_coords = [
        cube1_coords,
        cube2_coords,
        cube3_coords,
        cube4_coords,
        cube5_coords,
        cube6_coords,
    ]
    init_coords = cube_coords.copy()
    cube_approaches = [
        np.deg2rad([68.8, -68.3, 84.2, -107.1, -90, -18]),
        np.deg2rad([87.5, -45.5, 47.7, -102, -90.6, 3.3]),
        np.deg2rad([79.7, -46.9, 69.7, -105.1, -92.6, -10.1]),
        np.deg2rad([97.6, -38.3, -52.1, -100.8, -90.1, 8.5]),
        np.deg2rad([104.6, -85.3, 87.7, -90.5, -88.3, 17]),
        np.deg2rad([78.3, -61.7, 120.9, -87.6, -12.9, 27.7]),
    ]
    cubes_actual = [
        np.deg2rad([69, -63, 85, -107.7, -91.3, -18.2]),
        np.deg2rad([86.9, -40.1, 47.7, -102, -90.6, 3.3]),
        np.deg2rad([80.2, -43.4, 68.9, -107.1, -93.9, -9.4]),
        np.deg2rad([97.6, -34.6, 52.3, -100.8, -90.1, 8.5]),
        np.deg2rad([105.1, -86.3, 97.4, -102, -89.8, 19.3]),
        np.deg2rad([78, -56.6, 120.9, -93, -12.7, 29.4]),
    ]
    """
    x_lim = -0.34, 0.05
    y_lim = -0.55, -0.19
    """
    target_cube_coords = [
        [-0.205, -0.475, 0],
        [-0.115, -0.475, 0],
        [-0.115, -0.415, 0],
        [-0.205, -0.355, 0],
        [-0.115, -0.355, 0],
        [-0.115, -0.265, 0],
    ]
    try:
        target_cube_configs = np.load("palce_cube_configs.npy")
    except:
        target_cube_configs = [find_optimal_config(cube_approaches[cube_idx], target_cube_coords[cube_idx], cube_coords) for cube_idx in range(6)]
        np.save("palce_cube_configs.npy", target_cube_configs)

    if False:
    # debug: use sns to draw the cubes on a grid, as spheres with radius 0.1
        print(target_cube_coords)
        sns.set_theme()
        sns.set_style("whitegrid")
        sns.scatterplot(x=[cube[0] for cube in target_cube_coords], y=[cube[1] for cube in target_cube_coords], s=100)
        # draw lines for xlim, ylim
        plt.axhline(y=-0.55, color='r', linestyle='--')
        plt.axhline(y=-0.19, color='r', linestyle='--')
        plt.axvline(x=-0.34, color='r', linestyle='--')
        plt.axvline(x=0.05, color='r', linestyle='--')
        plt.show()
        exit()

    # for debugging
    if True:
        # load existing plans:
        existing_plans = []
        existing_plans_quality = [[np.inf, np.inf] for i in range(6)]
        try:
            for cube_idx in range(len(cube_coords)):
                grip_plan = np.load(f"grip_cube_{cube_idx+1}.npy")
                grip_plan_quality = get_plan_quality(grip_plan[:-1])
                place_plan = np.load(f"place_cube_{cube_idx+1}.npy")
                place_plan_quality = get_plan_quality(place_plan)
                existing_plans.append([grip_plan, place_plan])
                existing_plans_quality.append([grip_plan_quality, place_plan_quality])
        except:
            pass
        plans = []
        current_config = home
        import multiprocessing as mp
        pool = mp.Pool(12)
        for cube_idx in range(5, len(cube_coords)):
            path_quality = lambda p: len(p[0]) * p[1] if p[0] is not None else np.inf
            # optimize plan to grip cube
            results = pool.starmap(find_config_plan, [(current_config, cube_approaches[cube_idx], cube_coords)] * 128)
            grip_plan, min_cost = min(results, key=path_quality)
            if get_plan_quality(grip_plan) < existing_plans_quality[cube_idx][0]:
                print(f"[EXISTING] grip cube {cube_idx + 1} quality: {existing_plans_quality[cube_idx][0]:.3f}")
                print(f"[NEW] grip cube {cube_idx + 1} quality: {get_plan_quality(grip_plan):.3f}")
                print(f"grip min cost: {min_cost:.3f}, path length: {len(grip_plan)}")
                print()
                np.save(f"grip_cube_{cube_idx+1}", np.vstack((grip_plan, cubes_actual[cube_idx])))
            else:
                print(f"using existing path with quality {existing_plans_quality[cube_idx][0]:.3f} for grip cube {cube_idx + 1}")
                print()
            current_config = cube_approaches[cube_idx]
            
            # optimize plan to place cube
            cube_coords.pop(cube_idx)
            target_config = target_cube_configs[cube_idx]
            results = pool.starmap(find_config_plan, [(current_config, target_config, cube_coords)] * 48)
            place_plan, min_cost = min(results, key=path_quality)
            if get_plan_quality(place_plan) < existing_plans_quality[cube_idx][1]:
                print(f"[EXISTING] place cube {cube_idx + 1} quality: {existing_plans_quality[cube_idx][1]:.3f}")
                print(f"[NEW] place cube {cube_idx + 1} quality: {get_plan_quality(place_plan):.3f}")
                print(f"place min cost: {min_cost:.3f}, path length: {len(place_plan)}")
                print()
                np.save(f"place_cube_{cube_idx+1}", place_plan)
            else:
                print(f"using existing path with quality {existing_plans_quality[cube_idx][1]:.3f} for place cube {cube_idx + 1}")
                print()
            cube_coords.insert(cube_idx, target_cube_coords[cube_idx])
            try:
                grip_plan = np.vstack((grip_plan, cubes_actual[cube_idx]))
            except:
                grip_plan = []
            plans.append([grip_plan, place_plan])
        
    if False:
        env = Environment(env_idx=3)
        bb = Building_Blocks(transform, ur_params, env)
        plans = []
        for cube_idx in range(len(cube_coords)):
            grip_plan = np.load(f"grip_cube_{cube_idx+1}.npy")
            place_plan = np.load(f"place_cube_{cube_idx+1}.npy")
            plans.extend([grip_plan, place_plan])
        # generate list of current cube positions for each plan in plans
        cube_positions = []
        cubes = init_coords.copy()
        for cube_idx in range(6):
            cube_positions.append(cubes.copy())
            cubes.pop(cube_idx)
            cube_positions.append(cubes.copy())
            cubes.insert(cube_idx, target_cube_coords[cube_idx])

        gif_vis = VisualizeGif(ur_params, env, transform, bb)
        gif_vis.save_paths_to_gif(plans, cube_positions)
        

    # full_plan = np.concatenate(plans)
    # visualizer = Visualize_UR(ur_params, env, transform, bb)
    # print(plans[0][0])
    # visualizer.save_path_as_gif(plans[0][0], "plan1.gif")
