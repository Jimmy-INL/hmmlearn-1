import matplotlib.pyplot as plt
import pylab
import re
import torch
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import shutil
from PIL import Image
import warnings
from scipy.spatial.transform import Rotation as R
import pandas as pd


'''
This is a script I used to train the networks that map from xyz coordinates and quaternions to x,y center of mass
and height/width of the bounding box
'''


class BboxDataset(Dataset):
    def __init__(self, positions, bboxes):
        self.positions = positions
        self.bboxes = bboxes

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = torch.FloatTensor(self.positions[idx])
        bbox = torch.FloatTensor(self.bboxes[idx])
        return [position, bbox]


class SizeNetwork(nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super(SizeNetwork, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class PositionNetwork(nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super(PositionNetwork, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class NetworkBuilder(nn.Module):
    def __init__(self, state_dict):
        super(NetworkBuilder, self).__init__()
        num_layers = len([key for key in state_dict.keys() if 'weight' in key])
        hidden_size, input_size = state_dict['fc1.weight'].shape
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.layers = layers
        weights = sorted([key for key in state_dict.keys() if 'weight' in key])
        biases = sorted([key for key in state_dict.keys() if 'bias' in key])
        for idx, (weight, bias) in enumerate(zip(weights, biases)):
            self.layers[idx].state_dict()['weight'].data.copy_(state_dict[weight])
            self.layers[idx].state_dict()['bias'].data.copy_(state_dict[bias])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class Trainer(object):
    def __init__(self, hidden, batch_size, object_type, epochs, task, get_data=True):
        self.ROOT = '/Users/julianalverio/code/fetch_gym/models_and_data/'
        if not os.path.isdir(self.ROOT):
            self.ROOT = '/storage/jalverio/fetch_gym/models_and_data/'
        self.hidden = hidden
        self.object_type = object_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.prefix = os.path.join(self.ROOT, '%s_training/' % object_type)
        self.task = task
        assert self.task in ('size', 'position')
        if get_data:
            self.train_loader, self.test_loader = self.get_data()
        self.criterion = nn.MSELoss()

    def draw_box(self, image, left, right, top, bottom):
        top_row = [(top, x) for x in range(left, right+1)]
        bottom_row = [(bottom, x) for x in range(left, right + 1)]
        left_column = [(y, left) for y in range(top, bottom + 1)]
        right_column = [(y, right) for y in range(top, bottom + 1)]
        all_pixels = list(set(top_row + bottom_row + left_column + right_column))
        for y, x in all_pixels:
            image[y, x] = [0, 0, 0]
        return image

    def get_data(self):
        cube_files = os.listdir(self.prefix)
        coord_files = [os.path.join(self.prefix, file) for file in cube_files if '_v1' in file]
        all_positions = []
        all_bboxes = []
        for coord_file in coord_files:
            with open(coord_file) as f:
                position, bbox = f.readlines()
                position = self.array_from_text(position)
                bbox = self.array_from_text(bbox)
                if self.task == 'size' and self.object_type == 'cube':
                    # euler conversion
                    xyz_position = position[:3]
                    quat = position[3:]
                    euler = R.from_quat(quat).as_euler('xyz')
                    position = np.concatenate([xyz_position, euler], axis=0)
                else:
                    position = position[:3]
                all_positions.append(position)
                all_bboxes.append(bbox)
        all_positions = np.stack(all_positions, axis=0)
        all_bboxes = np.stack(all_bboxes, axis=0)

        train_positions = all_positions[:400]
        test_positions = all_positions[400:]
        train_bboxes = all_bboxes[:400]
        test_bboxes = all_bboxes[400:]

        train_dataset = BboxDataset(train_positions, train_bboxes)
        test_dataset = BboxDataset(test_positions, test_bboxes)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return train_loader, test_loader

    def evaluate_position(self):
        with torch.no_grad():
            total_loss = 0
            total_examples = 0
            for pos, lrtb in self.test_loader:
                preds = self.network(pos[:, :3])
                preds = torch.clamp(preds, 0, 499).round()
                center_x = (lrtb[:, 0] + lrtb[:, 1]) / 2.
                center_y = (lrtb[:, 2] + lrtb[:, 3]) / 2.
                target = torch.stack([center_x, center_y], dim=1)
                loss = self.criterion(preds, target)
                total_loss += loss.sum(dim=0)
                total_examples += pos.shape[0]
            mean_loss = (total_loss / total_examples).item()
            return mean_loss

    def evaluate_size(self):
        with torch.no_grad():
            total_loss = 0
            total_examples = 0
            for pos, lrtb in self.test_loader:
                preds = self.network(pos)
                preds = torch.clamp(preds, 0, 499).round()
                height = lrtb[:, 3] - lrtb[:, 2]
                width = lrtb[:, 1] - lrtb[:, 0]
                target = torch.stack([height, width], dim=1)
                loss = self.criterion(preds, target)
                total_loss += loss.sum(dim=0)
                total_examples += pos.shape[0]
            mean_loss = (total_loss / total_examples).item()
            return mean_loss

    def train(self):
        if self.task == 'size':
            self.network = SizeNetwork(6, self.hidden, 2)
        else:
            self.network = PositionNetwork(3, self.hidden, 2)
        self.optimizer = Adam(self.network.parameters(), lr=3e-3)

        if self.task == 'position':
            self.train_predict_center()
        else:
            self.train_predict_size()

    def train_predict_center(self):
        save_dir = self.ROOT + 'mapping_runs/%s_bbox_runs' % self.object_type
        shutil.rmtree(save_dir, ignore_errors=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            writer = SummaryWriter(save_dir)

        best_epoch = 0
        lowest_loss = np.inf
        for epoch in range(self.epochs):
            for pos, lrtb in self.train_loader:
                preds = self.network(pos)
                center_x = (lrtb[:, 0] + lrtb[:, 1]) / 2.
                center_y = (lrtb[:, 2] + lrtb[:, 3]) / 2.
                target = torch.stack([center_x, center_y], dim=1)
                loss = self.criterion(preds, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 20 == 0:
                mean_test_location_loss = self.save()
                if mean_test_location_loss < lowest_loss:
                    lowest_loss = mean_test_location_loss
                    best_epoch = epoch
                writer.add_scalar('test/%s_loss' % self.task, mean_test_location_loss, epoch)

        print('best epoch: %s' % best_epoch)
        print('lowest loss: %s' % lowest_loss)

    def train_predict_size(self):
        save_dir = self.ROOT + 'mapping_runs/%s_bbox_runs' % self.object_type
        shutil.rmtree(save_dir, ignore_errors=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            writer = SummaryWriter(save_dir)

        best_epoch = 0
        lowest_loss = np.inf
        for epoch in range(self.epochs):
            for pos, lrtb in self.train_loader:
                preds = self.network(pos)
                height = lrtb[:, 3] - lrtb[:, 2]
                width = lrtb[:, 1] - lrtb[:, 0]
                target = torch.stack([height, width], dim=1)
                loss = self.criterion(preds, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 20 == 0:
                mean_test_size_loss = self.save()
                if mean_test_size_loss < lowest_loss:
                    lowest_loss = mean_test_size_loss
                    best_epoch = epoch
                writer.add_scalar('test/loss', mean_test_size_loss, epoch)

        print('best epoch: %s' % best_epoch)
        print('lowest loss: %s' % lowest_loss)

    def save(self):
        if self.task == 'size':
            mean_test_location_loss = self.evaluate_size()
        else:
            mean_test_location_loss = self.evaluate_position()
        save_dir = self.ROOT + '%s_%s_networks/' % (self.object_type, self.task)
        paths = os.listdir(save_dir)
        if not paths:
            global_lowest_loss = 1.
        else:
            try:
                global_lowest_loss = min([float(path.split('_')[1]) for path in paths])
            except:
                import pdb; pdb.set_trace()
        if mean_test_location_loss < global_lowest_loss:
            if not paths:
                idx = 0
            else:
                idx = max([int(path.split('_')[3].split('.')[0]) for path in paths]) + 1
            file_name = 'loss_%s_idx_%s.torch' % (mean_test_location_loss, idx)
            torch.save(self.network.state_dict(), save_dir + file_name)
            print('saved', save_dir + file_name)
        return mean_test_location_loss

    def view_images_center(self):
        for image_idx in range(500):
            text_file = self.prefix + '%s_v1.txt' % image_idx
            image_file = self.prefix + '%s.npy' % image_idx
            image = np.load(image_file)
            with open(text_file) as f:
                position, bbox = f.readlines()
                with torch.no_grad():
                    position = np.array([float(val) for val in re.sub('[\n\[\]]', '', position).split('=')[1].split()])
                    xyz_position = torch.FloatTensor(position)
                    predicted_center = torch.clamp(self.network(xyz_position), 0, 499).round()
                    predicted_center = predicted_center.detach().clamp(0, 499).round().numpy().tolist()
                correct_bbox = np.array(eval(bbox.split('=')[1]))
                correct_width = correct_bbox[1] - correct_bbox[0]
                correct_height = abs(correct_bbox[3] - correct_bbox[2])
                predicted_l = max(predicted_center[0] - correct_width / 2, 0)
                predicted_r = min(predicted_center[0] + correct_width / 2, 499)
                predicted_t = max(predicted_center[1] - correct_height / 2, 0)
                predicted_b = min(predicted_center[1] + correct_height / 2, 499)
                predicted_bbox = [predicted_l, predicted_r, predicted_t, predicted_b]
                predicted_bbox = [int(round(x)) for x in predicted_bbox]
            # new_image = draw_box(image, *correct_bbox)
            new_image = self.draw_box(image, *predicted_bbox)
            save_path = '/Users/julianalverio/code/fetch_gym/models_and_data/hand_position_annotated_images/%s.png' % image_idx
            Image.fromarray(new_image).save(save_path, 'PNG')
            
            #Image.fromarray(new_image).show()
            

    def view_images_size(self):
        for image_idx in range(500):
            text_file = self.prefix + '%s_v1.txt' % image_idx
            image_file = self.prefix + '%s.npy' % image_idx
            image = np.load(image_file)
            with open(text_file) as f:
                position, bbox = f.readlines()
                l, r, t, b = eval(bbox.replace('\n', '').split('=')[1])
                correct_x = (l + r) / 2
                correct_y = (t + b) / 2
                with torch.no_grad():
                    xyz_quat = np.array(eval(position.replace('\n', '').split('=')[1]))
                    xyz_position = xyz_quat[:3]
                    quat = xyz_quat[3:]
                    euler = R.from_quat(quat).as_euler('xyz')
                    position = np.concatenate([xyz_position, euler], axis=0)
                    position = torch.FloatTensor(position)
                    predicted_size = torch.clamp(self.network(position), 0, 499).round()
                    predicted_size = predicted_size.detach().numpy().tolist()
                    predicted_height, predicted_width = predicted_size
                predicted_l = max(correct_x - predicted_width / 2, 0)
                predicted_r = min(correct_x + predicted_width / 2, 499)
                predicted_t = max(correct_y - predicted_height / 2, 0)
                predicted_b = min(correct_y + predicted_height / 2, 499)
                predicted_bbox = [predicted_l, predicted_r, predicted_t, predicted_b]
                predicted_bbox = [int(round(x)) for x in predicted_bbox]
            new_image = self.draw_box(image, *predicted_bbox)
            Image.fromarray(new_image).save('/Users/julianalverio/Desktop/%s_size_annotated_images/%s.png' % (self.object_type, image_idx))
            # Image.fromarray(new_image).show()
            # import pdb; pdb.set_trace()

    def array_from_text(self, text):
        array_text = re.sub('(.*=)', '', text)
        clean_text = re.sub('[\n\[\]]', '', array_text)
        split_text = [val for val in re.split('[,\s]', clean_text) if val]
        float_text = [float(val) for val in split_text]
        return np.array(float_text)
        
    def plot_cube_accuracy_size(self, size_network_path):
        cube_training_path = os.path.join(self.ROOT, 'hand_training')  # models_and_data
        network_path = os.path.join(self.ROOT, 'best_networks', size_network_path)
        network_state_dict = torch.load(network_path)
        network = NetworkBuilder(network_state_dict)
        height_errors = list()
        width_errors = list()
        for image_idx in range(500):
            label_file = os.path.join(self.ROOT, 'cube_training', '%s_v1.txt' % image_idx)
            with open(label_file, 'r+') as f:
                xyz_quat, lrtb = f.readlines()
                xyz_quat = self.array_from_text(xyz_quat)
                lrtb = self.array_from_text(lrtb)
                quat = xyz_quat[3:]
                xyz_position = xyz_quat[:3]
                euler = R.from_quat(quat).as_euler('xyz')
                network_input = torch.FloatTensor(np.concatenate([xyz_position, euler]))
                pred_height, pred_width = network(network_input)
                pred_height = round(pred_height.item())
                pred_width = round(pred_width.item())
                left, right, top, bottom = lrtb.tolist()
                left = int(left)
                right = int(right)
                top = int(top)
                bottom = int(bottom)
                label_center_x = int(round(np.mean([left, right])))
                label_center_y = int(round(np.mean([top, bottom])))
                label_height = abs(top - bottom)
                label_width = abs(right - left)
                pred_left = int(round(label_center_x - pred_width / 2))
                pred_right = int(round(label_center_x - pred_width / 2))
                pred_top = int(round(label_center_y - pred_height / 2))
                pred_bottom = int(round(label_center_y + pred_height / 2))
                height_error = abs(pred_height - label_height)
                width_error = abs(pred_width - label_width)
                height_errors.append(height_error)
                if width_error >= 3 or height_error >= 3:
                    path = os.path.join(self.ROOT, 'high_error_cube_size/%s_err%s_%s.png' % (image_idx, height_error, width_error))
                    old_path = os.path.join(self.ROOT, 'cube_training/%s.npy' % image_idx)
                    image = np.load(old_path)
                    image = self.draw_box(image, left, right, top, bottom)
                    image = self.draw_box(image, pred_left, pred_right, pred_top, pred_bottom)
                    Image.fromarray(image).save(path, 'PNG')
                    width_errors.append(width_error)
        height_df = pd.DataFrame(np.array(height_errors))
        width_df = pd.DataFrame(np.array(width_errors))
        height_plot = height_df.hist(bins=10)
        plt.savefig(os.path.join(self.ROOT, 'cube_height_errors'))
        plt.show()        
        width_plot = width_df.hist(bins=10)
        plt.savefig(os.path.join(self.ROOT, 'cube_width_errors'))
        plt.show()


    def plot_cube_accuracy_position(self, position_network_path):
        cube_training_path = os.path.join(self.ROOT, 'hand_training_test')  # models_and_data
        network_path = os.path.join(self.ROOT, 'best_networks', position_network_path)
        network_state_dict = torch.load(network_path)
        network = NetworkBuilder(network_state_dict)
        x_errors, y_errors = list(), list()
        for image_idx in range(500):
            label_file = os.path.join(self.ROOT, 'cube_training', '%s_v1.txt' % image_idx)
            with open(label_file, 'r+') as f:
                xyz_quat, lrtb = f.readlines()
                xyz_quat = self.array_from_text(xyz_quat)
                lrtb = self.array_from_text(lrtb)
                xyz_position = xyz_quat[:3]
                network_input = torch.FloatTensor(xyz_position)
                left, right, top, bottom = lrtb.tolist()
                label_center_x = int(round(np.mean([left, right])))
                label_center_y = int(round(np.mean([top, bottom])))
                label_height = int(round(abs(top - bottom)))
                label_width = int(round(abs(left - right)))
                left = int(round(left))
                right = int(round(right))
                top = int(round((top)))
                bottom = int(round(bottom))

                pred_x, pred_y = network(network_input)
                pred_x = round(pred_x.item())
                pred_y = round(pred_y.item())
                pred_left = int(round(pred_x - label_width/2.))
                pred_right = int(round(pred_x + label_width/2.))
                pred_top = int(round(pred_y - label_height/2.))
                pred_bottom = int(round(pred_y + label_height/2.))

                x_error = int(round(abs(label_center_x - pred_x)))
                y_error = int(round(abs(label_center_y - pred_y)))
                x_errors.append(x_error)
                y_errors.append(y_error)
                if x_error >= 3 or y_error >= 3:
                    path = os.path.join(self.ROOT, 'high_error_cube_position/%s_err%s_%s.png' % (image_idx, x_error, y_error))
                    old_path = os.path.join(self.ROOT, 'cube_training/%s.npy' % image_idx)
                    image = np.load(old_path)
                    image = self.draw_box(image, left, right, top, bottom)
                    image = self.draw_box(image, pred_left, pred_right, pred_top, pred_bottom)
                    Image.fromarray(image).save(path, 'PNG')
        x_df = pd.DataFrame(np.array(x_errors))
        y_df = pd.DataFrame(np.array(y_errors))
        x_plot = x_df.hist(bins=6)
        plt.savefig(os.path.join(self.ROOT, 'cube_x_position_errors'))
        plt.show()
        width_plot = y_df.hist(bins=7)
        plt.savefig(os.path.join(self.ROOT, 'cube_y_position_errors'))
        plt.show()

                

    # you have to modify this method every time you want to use it
    def view_images_size_and_location(self, size_network_path, position_network_path):
        size_network_path = os.path.join(self.ROOT, size_network_path)
        position_network_path = os.path.join(self.ROOT, position_network_path)
        size_network_state_dict = torch.load(size_network_path)
        position_network_state_dict = torch.load(position_network_path)
        size_hidden = size_network_state_dict['fc1.weight'].shape[0]
        position_hidden = position_network_state_dict['fc1.weight'].shape[0]
        size_network = SizeNetwork(3, size_hidden, 2)
        size_network.load_state_dict(size_network_state_dict)
        position_network = PositionNetwork(3, position_hidden, 2)
        position_network.load_state_dict(position_network_state_dict)

        for image_idx in range(500):
            text_file = self.prefix + '%s_v1.txt' % image_idx
            image_file = self.prefix + '%s.npy' % image_idx
            if image_idx == 407:
                try:
                    np.load(image_file)
                except FileNotFoundError:
                    continue
            image = np.load(image_file)
            with open(text_file) as f:
                position, bbox = f.readlines()
                # l, r, t, b = eval(bbox.replace('\n', '').split('=')[1])
                with torch.no_grad():
                    position = np.array(eval(position.replace('\n', '').split('=')[1]))
                    if self.object_type == 'cube':
                        xyz_position = position[:3]
                        quat = position[3:]
                        euler = R.from_quat(quat).as_euler('xyz')
                        position = np.concatenate([xyz_position, euler], axis=0)
                    position = torch.FloatTensor(position)
                    predicted_size = torch.clamp(size_network(position), 0, 499).round()
                    predicted_size = predicted_size.detach().numpy().tolist()
                    predicted_height, predicted_width = predicted_size

                    predicted_center = torch.clamp(position_network(torch.FloatTensor(position)), 0, 499).round()
                    predicted_x, predicted_y = predicted_center.detach().clamp(0, 499).round().numpy().tolist()

                predicted_l = max(predicted_x - predicted_width / 2, 0)
                predicted_r = min(predicted_x + predicted_width / 2, 499)
                predicted_t = max(predicted_y - predicted_height / 2, 0)
                predicted_b = min(predicted_y + predicted_height / 2, 499)
                predicted_bbox = [predicted_l, predicted_r, predicted_t, predicted_b]
                predicted_bbox = [int(round(x)) for x in predicted_bbox]
            new_image = self.draw_box(image, *predicted_bbox)
            Image.fromarray(new_image).save('/Users/julianalverio/Desktop/%s_size_and_position_annotated_images/%s.png' % (self.object_type, image_idx))
            # Image.fromarray(new_image).show()
            # import pdb; pdb.set_trace()


if __name__ == '__main__':
    hidden = 400
    batch_size = 128
    epochs = 2200
    object_type = 'hand'
    task = 'position'
    trainer = Trainer(hidden, batch_size, object_type, epochs, task, get_data=True)
    for _ in range(20):
        trainer.train()
    import sys; sys.exit()

    # trainer.plot_cube_accuracy_position('cube_position_network.torch')

    state_dict = torch.load('/Users/julianalverio/code/fetch_gym/models_and_data/hand_position_networks/loss_0.023599999025464058_idx_48.torch')
    network_builder = NetworkBuilder(state_dict)
    trainer.network = network_builder
    # torch.save(network_builder, '/Users/julianalverio/Desktop/best_networks/cube_size_network_builder.torch')

    # trainer.network.load_state_dict(state_dict)
    # trainer.network(torch.zeros([1, 3]))
    # torch.save(trainer.network, '/Users/julianalverio/Desktop/best_networks/cube_position_network_canned.torch')
    # import pdb; pdb.set_trace()

    # trainer.view_images_size_and_location()
    # for _ in range(20):
    #     trainer.train()

    # trainer.view_images_size_and_location(size_network_path='hand_size_networks/loss_0.13871033489704132_idx_16.torch', position_network_path='hand_position_networks/loss_0.0855051651597023_idx_62.torch')

    # train_predict_center(trainer.train_loader, trainer.test_loader, 'cube', 3, 100, 2, 1200, trainer.network, trainer.optimizer, trainer.criterion)


    # # size_network = train_predict_size(train_dataloader, test_dataloader, object_type, in_size, hidden, out_size, 1)
    # size_network = Network(in_size, hidden, out_size)
    # state_dict = torch.load('/Users/julianalverio/Desktop/hand_size_networks/loss_0.13871033489704132_idx_16.torch')
    # size_network.load_state_dict(state_dict)
    #
    # position_network = PositionNetwork(3, 100, 2)
    # state_dict = torch.load('/Users/julianalverio/Desktop/cube_location_networks/loss_0.18976563215255737_idx_41.torch')
    # position_network.load_state_dict(state_dict)
    #
    #
    # # view_images_size(size_network, prefix, object_type)
    # view_images_size_and_location(size_network, position_network, prefix, object_type)
