import sys
sys.path.insert(0, '/storage/jalverio/fetch_gym')
import gym
import pdb; pdb.set_trace()
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.
import numpy as np
import pickle
from PIL import Image
import os
from np.random import normal, uniform, choice


env = gym.make('FetchPickAndPlace-v1', reward_type='visual', reward_scale=5, dir_name='')
env.reset()

def generate_pickup_dataset():
    video_counter = 0
    while video_counter < 1000:
        print(video_counter)
        env.reset()
        success = generate_pickup_trajectory(video_counter)
        video_counter += int(success)

def get_noisy_action(action):
    noise = normal(loc=action, std=0.01)
    return action + noise

def constrain_target(target):
    target = target.copy()
    target[0] = max(target[0], 1.05)
    target[0] = min(target[0], 1.55)
    target[1] = max(target[1], 0.4)
    target[1] = min(target[1], 1.05)
    target[2] = max(target[2], 0.45)
    return target

def noisy_move_to_position(target, target_fingers, precision=None, random_fingers=False):
    observations = list()
    for _ in range(25):
        vector_to_target = target - env.get_robot_position()
        unit_vector_to_target = vector_to_target / np.linalg.norm(vector_to_target)
        magnitude = np.random.uniform(low=-0.03, high=0.03)
        action = unit_vector_to_target * magnitude
        noisy_action = get_noisy_action(action)
        finger_action = choice([0, 1]) if random_figners else target_fingers
        env.step(noisy_action)
        observations.append(env.get_obs())
        if precision:
            difference = np.linalg.norm(env.get_robot_position() - target)
            if difference <  precision:
                break
    return observations
        
#working
def generate_pickup_trajectory(video_counter):
    all_data = list()

    # random jitters to start
    num_jitters = round(abs(normal(loc=0, scale=2)))
    for jitter in range(num_jitters):
        action = uniform(low=-0.03, high=0.03, size=4)
        env.step(action)

    # move to a position within a large cube around the block
    object_position = env.get_object_position()
    target_x = uniform(low=object_position[0] - 0.1, high=object_position[0] + 0.1)
    target_y = uniform(low=object_position[1] - 0.1, high=object_position[1] + 0.1)
    target_z = uniform(low=object_position[2] - 0.1, high=object_position[2] + 0.1)
    target_width = choice([0, 1])
    noisy_move_to_position([target_x, target_y, target_z], target_width, random_fingers=True, precision=0.1)

    # move to a position within a small cube around the block (maybe bump into it)
    object_position = env.get_object_position()
    target_x = uniform(low=object_position[0] - 0.05, high=object_position[0])
    target_y = uniform(low=object_position[1] - 0.05, high=object_position[1])
    target_z = uniform(low=object_position[2] - 0.05, high=object_position[2])
    target_width = 1
    noisy_move_to_target([target_x, target_y, target_z], target_width, random_fingers=False, precision=0.05)

    

    
    
                       
    

    
    all_data = list()
    robot_position = env.get_robot_position()
    object_position_quat = env.sim.data.get_joint_qpos('object0:joint')
    object_starting_z = object_position_quat[2].copy()
    finger_width = env.get_finger_width()

    frame = env.render(mode='rgb_array')
    all_data.append((frame.copy(), robot_position.copy(), object_position_quat.copy(), finger_width))

    x = np.random.uniform(1.1, 1.5)
    y = np.random.uniform(0.45, 1.05)
    z = np.random.uniform(0.5, 0.75)
    random_start_position = np.array([x, y, z])
    _, frame_data = env.move(random_start_position, return_frames=True, precision_threshold=0.05, fingers=True)
    all_data.extend(frame_data)

    x_noise = np.random.uniform(low=-0.03, high=0.03)
    y_noise = np.random.uniform(low=-0.03, high=0.03)
    z_noise = np.random.uniform(low=0.04, high=0.08)
    object_position_quat = env.sim.data.get_joint_qpos('object0:joint')
    object_position = object_position_quat[:3]
    preparing_to_grab = object_position + np.stack([x_noise, y_noise, z_noise])
    _, frame_data = env.move(preparing_to_grab, return_frames=True, gripper=1, precision_threshold=0.05, stability_threshold=1, fingers=True)
    all_data.extend(frame_data)

    x1_noise = np.random.uniform(low=-0.03, high=0.03)
    y1_noise = np.random.uniform(low=-0.03, high=0.03)
    z1_noise = np.random.uniform(low=0.0, high=0.03)
    object_position_quat = env.sim.data.get_joint_qpos('object0:joint')
    object_position = object_position_quat[:3]
    half_ready_to_grab = object_position + np.array([x1_noise, y1_noise, 0.03])
    _, frame_data = env.move(half_ready_to_grab, return_frames=True, stability_threshold=1, precision_threshold=0.01, fingers=True)
    all_data.extend(frame_data)

    object_position_quat = env.sim.data.get_joint_qpos('object0:joint')
    object_position = object_position_quat[:3]
    ready_to_grab = object_position + np.array([x1_noise, y1_noise, z1_noise])
    _, frame_data = env.move(ready_to_grab, return_frames=True, gripper=-1, stability_threshold=0, fingers=True)
    for _ in range(5):
        env.step_postprocessed([0., 0., 0., -1.])

    x2 = np.random.uniform(1.1, 1.5)
    y2 = np.random.uniform(0.45, 1.05)
    z2 = np.random.uniform(0.5, 0.85)
    random_end_position = np.array([x2, y2, z2])
    _, frame_data = env.move(random_end_position, return_frames=True, gripper=-1, fingers=True)
    all_data.extend(frame_data)

    padding = 50 - len(all_data)
    for _ in range(padding):
        all_data.append(all_data[-1])

    # quality control
    for frame_data in all_data:
        assert len(frame_data) == 4
        frame, robot_position, object_position_quat, finger_width = frame_data
        assert frame.shape == (500, 500, 3)
        assert robot_position.shape == (3,)
        assert object_position_quat.shape == (7,)
        assert isinstance(finger_width, np.float64)

    # make sure the object actually got picked up
    end_object_z = env.sim.data.get_joint_qpos('object0:joint')[2]
    if end_object_z < (object_starting_z + 0.05):
        return False
    if env.get_finger_width() < 0.04:
        return False

    with open('/storage/jalverio/fetch_gym/models_and_data/pickup_videos/%s.pkl' % video_counter, 'wb') as f:
        pickle.dump(all_data, f)
    return True


def generate_approach_dataset():
    for video_counter in range(10):
        print(video_counter)
        env.reset()
        env.generate_approach_trajectory(video_counter)


def generate_approach_trajectory(env, video_counter):
    all_data = list()
    robot_position = env.sim.data.get_site_xpos('robot0:grip')
    object_position_quat = env.sim.data.get_joint_qpos('object0:joint')
    object_starting_z = object_position_quat[2].copy()

    frame = env.render(mode='rgb_array')
    all_data.append((frame.copy(), robot_position.copy(), object_position_quat.copy()))

    x = np.random.uniform(1.1, 1.5)
    y = np.random.uniform(0.45, 1.05)
    z = np.random.uniform(0.5, 0.75)
    random_start_position = np.array([x, y, z])
    _, frame_data = env.move(random_start_position, return_frames=True, precision_threshold=0.05)
    all_data.extend(frame_data)

    x_noise = np.random.uniform(low=-0.03, high=0.03)
    y_noise = np.random.uniform(low=-0.03, high=0.03)
    z_noise = np.random.uniform(low=0.04, high=0.08)
    object_position_quat = env.sim.data.get_joint_qpos('object0:joint')
    object_position = object_position_quat[:3]
    preparing_to_grab = object_position + np.stack([x_noise, y_noise, z_noise])
    _, frame_data = env.move(preparing_to_grab, return_frames=True, gripper=1, precision_threshold=0.05,
                              stability_threshold=1)
    all_data.extend(frame_data)

    x1_noise = np.random.uniform(low=-0.03, high=0.03)
    y1_noise = np.random.uniform(low=-0.03, high=0.03)
    z1_noise = np.random.uniform(low=0.0, high=0.03)
    object_position_quat = env.sim.data.get_joint_qpos('object0:joint')
    object_position = object_position_quat[:3]
    half_ready_to_grab = object_position + np.array([x1_noise, y1_noise, 0.03])
    _, frame_data = env.move(half_ready_to_grab, return_frames=True, stability_threshold=1, precision_threshold=0.01)
    all_data.extend(frame_data)

    object_position_quat = env.sim.data.get_joint_qpos('object0:joint')
    object_position = object_position_quat[:3]
    ready_to_grab = object_position + np.array([x1_noise, y1_noise, z1_noise])
    _, frame_data = env.move(ready_to_grab, return_frames=True, gripper=-1, stability_threshold=0)
    for _ in range(5):
        env.step([0, 0, 0, -1])

    x2 = np.random.uniform(1.1, 1.5)
    y2 = np.random.uniform(0.45, 1.05)
    z2 = np.random.uniform(0.5, 0.85)
    random_end_position = np.array([x2, y2, z2])
    _, frame_data = env.move(random_end_position, return_frames=True, gripper=-1)
    all_data.extend(frame_data)

    padding = 50 - len(all_data)
    for _ in range(padding):
        all_data.append(all_data[-1])

    # quality control
    for frame, robot_position, object_position_quat in all_data:
        assert frame.shape == (500, 500, 3)
        assert robot_position.shape == (3,)
        assert object_position_quat.shape == (7,)

    # make sure the object actually got picked up
    end_object_z = env.sim.data.get_joint_qpos('object0:joint')[2]
    if end_object_z < (object_starting_z + 0.05):
        return

    with open('/storage/jalverio/sentence_tracker/models_and_data/pickup_videos/%s.pkl' % video_counter, 'wb') as f:
        pickle.dump(all_data, f)


# format: [x,y,z,quat]
def generate_object_dataset():
    for _ in range(30):
        env.step([0, 0, 1, -1])
    prefix = '/storage/jalverio/fetch_gym/models_and_data/cube_training_test/'
    env.render(mode='rgb_array')
    for counter in range(500):
        # max x: 1.5, min x: 1.1
        # max y: 1.1, min y: 0.4
        # min z: 0.424, max z: 0.9
        x = np.random.uniform(1.1, 1.5)
        # y = np.random.uniform(0.55, 0.95)
        y_lower_bound = 0.25 * (x - 1.1) + 0.45
        y_upper_bound = 0.25 * (1.5 - x) + 0.95
        y = np.random.uniform(y_lower_bound, y_upper_bound)
        z = np.random.uniform(0.45, 0.75)
        pos = [x, y, z]
        quat = np.random.uniform(size=4)
        quat /= np.linalg.norm(quat)
        quat = quat.tolist()
        xpos = np.array(pos + quat)
        env.sim.data.set_joint_qpos('object0:joint', xpos)

        with open(os.path.join(prefix, '%s.txt' % counter), 'w+') as f:
            f.write('xyzquat=' + str(xpos))
        image = env.render(mode='rgb_array')
        np.save(os.path.join(prefix, '%s.npy' % counter), image)
        image = Image.fromarray(image)
        image.save(os.path.join(prefix, '%s.png' % counter), 'PNG')
        print(counter)


def generate_robot_dataset():
    prefix = '/storage/jalverio/fetch_gym/models_and_data/hand_training/'
    env.render(mode='rgb_array')
    image_idx = 0
    while image_idx < 500:
        env.reset()
        for _ in range(35):
            env.step(np.random.uniform(-1, 1, size=4))
        if np.random.uniform(0, 1) > 0.6:
            randomize_fingers()
        # pos = env.sim.data.get_site_xpos('robot0:grip')
        r_finger_pos = env.sim.data.get_body_xipos('robot0:r_gripper_finger_link')
        l_finger_pos = env.sim.data.get_body_xipos('robot0:l_gripper_finger_link')
        pos = (r_finger_pos + l_finger_pos) / 2
        max_x = max(r_finger_pos[0], l_finger_pos[0])
        min_x = min(r_finger_pos[0], l_finger_pos[0])
        max_y = max(r_finger_pos[1], l_finger_pos[1])
        min_y = min(r_finger_pos[1], l_finger_pos[1])
        max_z = max(r_finger_pos[2], l_finger_pos[2])
        min_z = min(r_finger_pos[2], l_finger_pos[2])
        if not (max_x < 1.478 and min_x > 1.08 and max_y < 0.957 and min_y > 0.52 and max_z < 0.806 and min_z > 0.449):
            continue
        image = env.render(mode='rgb_array')
        text = 'xyz=' + str(pos) + '\n'
        np.save(prefix + '%s.npy' % image_idx, image)
        Image.fromarray(image).save(prefix + '%s.png' % image_idx, 'PNG')
        with open(prefix + '%s.txt' % image_idx, 'w+') as f:
            f.write(text)
            f.flush()
        print(image_idx)
        image_idx += 1

def get_robot_position():
    r_finger_position = env.sim.data.get_body_xipos('robot0:r_gripper_finger_link')
    l_finger_position = env.sim.data.get_body_xipos('robot0:l_gripper_finger_link')
    mean_position = np.stack([l_finger_position, r_finger_position], axis=0).mean(axis=0)
    return mean_position

def test():
    for _ in range(6):
        env.step([11, 0, -1, 0])
    env.render(mode='human')
    save()
    import pdb; pdb.set_trace()
    

def save():
    Image.fromarray(env.render(mode='rgb_array')).save('/storage/jalverio/test.png', 'PNG')




if __name__ == '__main__':
    test()
    generate_pickup_dataset()
    # generate_object_dataset()
    # generate_robot_dataset()
