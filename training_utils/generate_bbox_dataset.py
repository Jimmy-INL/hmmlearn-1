import numpy as np
from PIL import Image
import os
from colorsys import rgb_to_hsv


def draw_box(image, left, right, top, bottom):
    top_row = [(top, x) for x in range(left, right+1)]
    bottom_row = [(bottom, x) for x in range(left, right + 1)]
    left_column = [(y, left) for y in range(top, bottom + 1)]
    right_column = [(y, right) for y in range(top, bottom + 1)]
    all_pixels = list(set(top_row + bottom_row + left_column + right_column))
    for y, x in all_pixels:
        image[y, x] = [0, 0, 0]
    return image

def get_object_bboxes(image):
    red_pixels = list()
    for y in range(500):
        for x in range(500):
            r, g, b = image[y, x]
            if r > b and r > g:
            # if b > 80 and r < 50 and g < 50:
                red_pixels.append((y, x))
                # image[y, x] = [255, 0, 0]
    all_x = [x for y, x in red_pixels]
    all_y = [y for y, x in red_pixels]
    right = max(all_x) + 1
    left = min(all_x) - 1
    bottom = max(all_y) + 1
    top = min(all_y) - 1
    # draw_box(image, left, right, top, bottom)
    return left, right, top, bottom


def get_hand_bboxes(image):
    red_pixels = list()
    for y in range(500):
        for x in range(500):
            r, g, b = image[y, x]
            if r > b and r > g:
                red_pixels.append((y, x))
                image[y, x] = [255, 0, 0]
    all_x = [x for y, x in red_pixels]
    all_y = [y for y, x in red_pixels]
    right = max(all_x) + 1
    left = min(all_x) - 1
    bottom = max(all_y) + 1
    top = min(all_y) - 1
    right = min(right, 499)
    left = max(left, 0)
    top = max(top, 0)
    bottom = min(bottom, 499)
    draw_box(image, left, right, top, bottom)
    return left, right, top, bottom


if __name__ == '__main__':
    # # TODO: alter this path as needed
    # prefix = '/Users/julianalverio/code/fetch_gym/models_and_data/cube_training_test/'
    # for file_idx in range(500):n
    #     path = os.path.join(prefix, '%s.npy' % file_idx)
    #     png_path = os.path.join(prefix, '%s_boxed.png' % file_idx)
    #     old_text_path = os.path.join(prefix, '%s.txt' % file_idx)
    #     new_text_path = os.path.join(prefix, '%s_v1.txt' % file_idx)
    #     image = np.load(path)
    #     assert 'hand' in prefix or 'cube' in prefix
    #     if 'cube' in prefix:
    #         left, right, top, bottom = get_object_bboxes(image)
    #     elif 'hand' in prefix:
    #         left, right, top, bottom = get_hand_bboxes(image)
    #     labeled_image = draw_box(image, left, right, top, bottom)
    #     Image.fromarray(labeled_image).save(png_path, 'PNG')
    #     with open(old_text_path) as f:
    #         lines = f.read().replace('\n', '').replace('[', '').replace(']', '')
    #         lines = lines.replace('xyzquat=', '').split(' ')
    #         lines = [float(number) for number in lines if number]
    #     text = 'xyzquat=' + str(lines) + '\n'
    #     text += 'lrtb=' + str([left, right, top, bottom])
    #     with open(new_text_path, 'w+') as f:
    #         f.write(text)
    #         f.flush()

    
    prefix = '/Users/julianalverio/code/fetch_gym/models_and_data/hand_training/'
    for file_idx in range(500):
        path = prefix + '%s.npy' % file_idx
        old_text_path = os.path.join(prefix, '%s.txt' % file_idx)
        new_text_path = os.path.join(prefix, '%s_v1.txt' % file_idx)
        image = np.load(path)
        object_left, object_right, object_top, object_bottom = get_hand_bboxes(image)
        with open(old_text_path) as f:
            text = f.read()
        Image.fromarray(image).save(os.path.join(prefix, '%s_boxed.png' % file_idx), 'PNG')
        text += 'lrtb=' + str([object_left, object_right, object_top, object_bottom])
        with open(new_text_path, 'w+') as f:
            f.write(text)
            f.flush()
        if file_idx % 50 == 0:
            print(file_idx)
