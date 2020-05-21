import sys
sys.path.insert(0, '/storage/jalverio/fetch_gym')
import pdb; pdb.set_trace()
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.
import numpy as np
import pickle
from PIL import Image
import os
from numpy.random import normal, uniform, choice

import pickle
import cv2
import os
import shutil
import re

sys.path.insert(0, '/storage/jalverio/fetch_gym/gym/envs/robotics/')
from robot_env import RobotEnv

class DatasetGenerator(object):
    def __init__(self):
        objects = [['red block', [0.8, 0.8]]]
        env = RobotEnv(reward_scale=5, dir_name='', objects=objects)
        env.reset()
        self.env = env

        self.all_frames = list()
        self.all_observations = list()
        self.frames = list()
        self.observations = list()
        self.root = '/storage/jalverio/hmmlearn/training_utils/pickup_dataset'
        self.previous_object_relative_position = None
        self.previous_robot_position = None
        self.previous_object_position = None

    def snapshot(self):
        self.frames.append(self.env.render(mode='rgb_array'))
        observation = self.env.get_obs()
        finger_width, object_rel_pos, object_rot, object_velp, object_velr, left_finger_rel, right_finger_rel, ready_to_close, distance, object_position, robot_position, fractional_distance, fractional_distances = observation
        obs_dict = dict()
        obs_dict['finger_width'] = finger_width
        obs_dict['object_relative_position'] = object_rel_pos
        obs_dict['object_rotation'] = object_rot
        obs_dict['object_velp'] = object_velp
        obs_dict['object_velr'] = object_velr
        obs_dict['left_finger_relative_position'] = left_finger_rel
        obs_dict['right_finger_relative_position'] = right_finger_rel
        obs_dict['ready_to_close'] = ready_to_close
        obs_dict['distance'] = distance
        obs_dict['robot_position'] = robot_position
        obs_dict['object_position'] = object_position
        obs_dict['fractional_distance'] = fractional_distance
        obs_dict['fractional_distances'] = fractional_distances

        # object relative velocity through object relative position
        if self.previous_object_relative_position is None:
            object_relative_velocity = np.zeros(3)
            robot_velocity = np.zeros(3)
            object_velocity = np.zeros(3)
        else:
            object_relative_velocity = obs_dict['object_relative_position'] - self.previous_object_relative_position
            robot_velocity = obs_dict['robot_position'] - self.previous_robot_position
            object_velocity = obs_dict['object_position'] - self.previous_object_position

        obs_dict['object_relative_velocity'] = object_relative_velocity
        obs_dict['robot_velocity'] = robot_velocity
        obs_dict['object_velocity'] = object_velocity
        
        self.previous_object_relative_position = obs_dict['object_relative_position']
        self.previous_robot_position = robot_position
        self.previous_object_position = object_position
        
        self.observations.append(obs_dict)

    def end_episode(self, success):
        if success:
            regex = r'\d+\.pkl'
            pkl_files = [path for path in os.listdir(self.root) if re.match(regex, path)]
            if not pkl_files:
                next_idx = 0
            else:
                next_idx = max([int(path.replace('.pkl', '')) for path in pkl_files]) + 1
            obs_save_path = os.path.join(self.root, '%s.pkl' % next_idx)
            with open(obs_save_path, 'wb') as f:
                pickle.dump(self.observations, f)
            mp4_save_path = os.path.join(self.root, '%s.mp4' % next_idx)
            self.write_mp4(mp4_save_path)
            frames_save_path = os.path.join(self.root, '%s_frames.pkl' % next_idx)
            with open(frames_save_path, 'wb') as f:
                pickle.dump(self.frames, f)
        self.observations = list()
        self.frames = list()

    def generate_pickup_dataset(self):
        fail_counter = 0
        video_counter = 0
        while video_counter < 600:
            self.env.reset()
            success = self.generate_pickup_trajectory()
            video_counter += int(success)
            if not success:
                fail_counter += 1
            print(video_counter, 'successes')
            print(fail_counter, 'fails \n')
            self.end_episode(success)


    def constrain_target(self, target):
        target = target.copy()
        target[0] = max(target[0], 1.05)
        target[0] = min(target[0], 1.55)
        target[1] = max(target[1], 0.4)
        target[1] = min(target[1], 1.05)
        target[2] = max(target[2], 0.45)
        return target

    # move from start to the goal, don't bump the block, finish within total_moves
    def interpolate_trajectory(self, target, finger_target, total_moves):
        for move in range(int(total_moves)):
            object_position = self.env.get_object_position()
            robot_position = self.env.get_robot_position()
            target_vector = object_position - robot_position
            if np.all(abs(target_vector) < 0.03):
                goal = target_vector
                finger_goal = finger_target
            else:
                velocity = uniform(low=0.001, high=0.03)
                target_unit_vector = target_vector / np.linalg.norm(target_vector)
                target_vector_clean = target_unit_vector * velocity
                noise = normal(loc=0, scale=0.01, size=3)
                goal = robot_position + target_vector_clean + noise
                finger_goal = choice([0, 1])

            goal = self.constrain_target(goal)
            if self.collision_check(goal, finger_goal):
                return move
            self.custom_move(goal, finger_goal, precision_threshold=0.01, attempts=25)
            self.snapshot()
        return total_moves

    # spend num_moves moving in the area around the block. Don't bump the block.
    def move_near_block(self, moves_remaining):
        while moves_remaining > 0:
            object_position = self.env.get_object_position()
            target = uniform(low=object_position - 0.1, high=object_position + 0.1)
            moves_used = self.interpolate_trajectory(target, 1, moves_remaining)
            moves_remaining -= moves_used

    # working
    # you generally don't want save frames! Only for debugging
    def custom_move(self, goal, finger_goal, stability_threshold=2, precision_threshold=0.005, attempts=20, save_frames=False):
        error = goal - self.env.get_robot_position()
        finger_error = np.array([finger_goal - self.env.get_finger_width()])
        error = np.concatenate([error, finger_error], axis=0)

        stability_counter = 0
        for attempt in range(attempts):
            output = self.env.controller.update(error)
            if finger_goal == 0:
                output[-1] = -1
            step_output = self.env._step_raw([*output])
            if save_frames:
                snapshot()

            error = goal - self.env.get_robot_position()
            finger_error = np.array([finger_goal - self.env.get_finger_width()])
            error = np.concatenate([error, finger_error], axis=0)
            if np.linalg.norm(error[:3]) < 0.005 and abs(finger_error) < 0.003:
                stability_counter += 1
            else:
                stability_counter = 0
            if stability_counter >= stability_threshold:
                return step_output
        return step_output


    # get totally aligned, ready to grab
    def align_with_block(self):
        current_position = self.env.get_robot_position()
        self.custom_move(current_position, 1)  # open fingers early
        self.snapshot()
        grab_height = uniform(0.45, 0.48)
        y_offset = uniform(-0.02, 0.02)
        x_offset = uniform(-0.05, 0.05)
        target_position = self.env.get_object_position()
        target_position[2] = grab_height
        target_position[1] += y_offset
        target_position[0] += x_offset

        current_position = self.env.get_robot_position()
        collision = False
        for weight in np.arange(0, 1, 0.1):
            new_position = current_position * weight + target_position * (1 - weight)
            if self.collision_check(new_position, 1):
                collision = True
        object_position = self.env.get_object_position()
        current_position = self.env.get_robot_position()
        if collision:
            fix_target = self.env.get_object_position()
            if target_position[0] > current_position[0]:
                fix_target[0] += 0.05
            else:
                fix_target[0] -= 0.05
            fix_target[2] += 0.03
            self.custom_move(fix_target, 1)
            self.snapshot()

        self.custom_move(target_position, 1)
        self.snapshot()

    # actually grab the block
    def grab_block(self):
        y_offset = uniform(-0.02, 0.02)
        x_offset = uniform(-0.02, 0.02)
        z_position = uniform(0.45, 0.48)

        target_position = self.env.get_object_position()
        target_position[0] += x_offset
        target_position[1] += y_offset
        target_position[2] = z_position
        self.custom_move(target_position, 1)
        self.snapshot()
        self.custom_move(target_position, 0)
        self.snapshot()

    # Once you're holding the block, drop it
    def drop_block(self):
        noise = uniform(low=-0.03, high=0.03, size=3)

        action.append(1)
        self.env.step(action)
        self.snapshot()

    def noisy_lift(self):
        for _ in range(25):
            target = self.env.get_robot_position()
            target[0] += uniform(-0.03, 0.03)
            target[1] += uniform(-0.03, 0.03)
            target[2] += uniform(low=-0.01, high=0.03)  # move upwards, in expectation

            self.custom_move(target, 0)
            self.snapshot()
            object_position = self.env.get_object_position()
            if object_position[2] > 0.624:
                break


    def random_movements(self, num_movements):
        for _ in range(num_movements):
            action = uniform(low=-0.03, high=0.03, size=4)
            self.env.step(action)
            self.snapshot()


    def write_mp4(self, path=None):
        if path is None:
            path = '/storage/jalverio/test.mp4'
        shape = (500, 500)
        writer = cv2.VideoWriter(path, 0x7634706d, 5, shape)
        for frame in self.frames:
            writer.write(frame)
        writer.release()

    # return true if reaching the fingers would collide with the block
    def collision_check(self, hand_target, finger_width):
        assert finger_width in [0, 1]
        object_position = self.env.get_object_position()
        object_x, object_y, object_z = object_position
        x_target, y_target, z_target = hand_target
        x_in_range = False
        y_in_range = False
        z_in_range = False

        y_difference = abs(y_target - object_y)
        if y_difference < 0.09:
            y_in_range = True
        z_difference = abs(z_target - object_z)
        if z_difference < 0.075:
            z_in_range = True
        x_difference = abs(x_target - object_x)
        if x_difference  < 0.045:
            x_in_range = True
        collision = x_in_range and y_in_range and z_in_range
        x_difference = round(x_difference, 2)
        y_difference = round(y_difference, 2)
        z_difference = round(z_difference, 2)
        return collision


    def generate_pickup_trajectory(self):
        self.snapshot()

        # random jitters to start
        num_jitters = round(abs(normal(loc=0, scale=5)))
        self.random_movements(num_jitters)
        # print(num_jitters, 'jitters')

        # move to a position within a large cube around the block
        moves_near_block = 25 - num_jitters
        # print(moves_near_block, 'moves near block')
        self.move_near_block(moves_near_block)

        # # prepare to grab
        self.align_with_block()
        self.grab_block()
        self.noisy_lift()

        self.write_mp4()

        if self.env.get_object_position()[2] > 0.54:
            print('success')
            return True
        print('failure')
        return False


    # # format: [x,y,z,quat]
    # def generate_object_dataset(self):
    #     for _ in range(30):
    #         self.env.step([0, 0, 1, -1])
    #     prefix = '/storage/jalverio/fetch_gym/models_and_data/cube_training_test/'
    #     self.env.render(mode='rgb_array')
    #     for counter in range(500):
    #         # max x: 1.5, min x: 1.1
    #         # max y: 1.1, min y: 0.4
    #         # min z: 0.424, max z: 0.9
    #         x = np.random.uniform(1.1, 1.5)
    #         # y = np.random.uniform(0.55, 0.95)
    #         y_lower_bound = 0.25 * (x - 1.1) + 0.45
    #         y_upper_bound = 0.25 * (1.5 - x) + 0.95
    #         y = np.random.uniform(y_lower_bound, y_upper_bound)
    #         z = np.random.uniform(0.45, 0.75)
    #         pos = [x, y, z]
    #         quat = np.random.uniform(size=4)
    #         quat /= np.linalg.norm(quat)
    #         quat = quat.tolist()
    #         xpos = np.array(pos + quat)
    #         env.sim.data.set_joint_qpos('object0:joint', xpos)

    #         with open(os.path.join(prefix, '%s.txt' % counter), 'w+') as f:
    #             f.write('xyzquat=' + str(xpos))
    #         image = env.render(mode='rgb_array')
    #         np.save(os.path.join(prefix, '%s.npy' % counter), image)
    #         image = Image.fromarray(image)
    #         image.save(os.path.join(prefix, '%s.png' % counter), 'PNG')
    #         print(counter)

    def test(self):
        object_position = self.env.get_object_position()
        target = object_position.copy()
        target[1] -= 0.045
        target[2] = 0.45
        self.custom_move(target, 0, save_frames=True)
        self.write_mp4()



    def save(self):
        Image.fromarray(self.env.render(mode='rgb_array')).save('/storage/jalverio/test.png', 'PNG')



if __name__ == '__main__':
    gen = DatasetGenerator()
    gen.generate_pickup_dataset()
