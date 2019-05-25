import os
import cv2
import math
import numpy as np
from util import timed


def process_annotation(image, output_path, annotation, offsets=None):
    major_axis_radius, minor_axis_radius, angle, center_x, center_y = annotation

    if abs(angle) < math.pi / 4:
        rx, ry = major_axis_radius, minor_axis_radius
    else:
        rx, ry = minor_axis_radius, major_axis_radius

    border_x = int(rx) * 4
    border_y = int(ry) * 4
    center_x += border_x
    center_y += border_y

    image_expanded = cv2.copyMakeBorder(
        image,
        border_y,  # up
        border_y,  # down
        border_x,  # left
        border_x,  # right
        cv2.BORDER_REPLICATE
    )

    box = list(map(int, [
        center_x - 4/3 * rx,  # left-up x
        center_y - 4/3 * ry,  # left-up y
        center_x + 4/3 * rx,  # right-down x
        center_y + 4/3 * ry,  # right-down y
    ]))

    if offsets is not None:
        box += np.int32(np.array(offsets) * np.array([rx, ry, rx, ry]))

    assert (box[0] > 0 and box[1] > 0 and box[2] > 0 and box[3] > 0 and
            box[3] > box[1] and box[2] > box[0])

    image_cropped = image_expanded[box[1]:box[3], box[0]:box[2]]
    cv2.imwrite(output_path, image_cropped)


def get_face(file):
    with open(file) as f:
        while True:
            image_name = f.readline().strip()
            if image_name == '':
                break
            image_path = os.path.join('data', 'originalPics', image_name + '.jpg')
            n_faces = int(f.readline())
            for _ in range(n_faces):
                annotation = list(map(float, f.readline().split()[:-1]))
                yield image_path, annotation


## slide on different directions
offsets = [
    [ 1,  0,  1,  0],
    [ 0,  1,  0,  1],
    [-1,  0, -1,  0],
    [ 0, -1,  0, -1],
    [ 1,  1,  1,  1],
    [-1, -1, -1, -1],
    [ 1, -1,  1, -1],
    [-1,  1, -1,  1]
]

for root, dirs, files in os.walk(os.path.join('data', 'FDDB-folds')):
    files = [os.path.join(root, file) for file in sorted(files)[::2]]

    if not os.path.exists(os.path.join('data', 'train')):
        os.makedirs(os.path.join('data', 'train'))
    if not os.path.exists(os.path.join('data', 'test')):
        os.makedirs(os.path.join('data', 'test'))

    with open('data/train.txt', 'w') as f:
        ## generate positive samples for training
        cnt = 0
        for file in files[:8]:  ## first 8 folders to train
            with timed(f'generate pos samples with {file} for training'):
                for image_path, annotation in get_face(file):
                    image = cv2.imread(image_path)
                    output_path = os.path.join('data', 'train', f'pos_{cnt}.jpg')
                    process_annotation(image, output_path, annotation)
                    f.write(f'{output_path} 1\n')
                    cnt += 1

        ## generate negative samples for training with the first four folders
        cnt = 0
        for file in files[:4]:
            with timed(f'generate neg samples with {file} for training'):
                for image_path, annotation in get_face(file):
                    image = cv2.imread(image_path)
                    for offset in offsets:
                        output_path = os.path.join('data', 'train', f'neg_{cnt}.jpg')
                        process_annotation(image, output_path, annotation, offset)
                        f.write(f'{output_path} 0\n')
                        cnt += 1

    with open('data/test.txt', 'w') as f:
        ## generate positive samples for testing
        cnt = 0
        for file in files[-2:]:  ## first 8 folders to train
            with timed(f'generate pos samples with {file} for testing'):
                for image_path, annotation in get_face(file):
                    image = cv2.imread(image_path)
                    output_path = os.path.join('data', 'test', f'pos_{cnt}.jpg')
                    process_annotation(image, output_path, annotation)
                    f.write(f'{output_path} 1\n')
                    cnt += 1

        ## generate negative samples for testing with the last folder
        cnt = 0
        for file in files[-1:]:
            with timed(f'generate neg samples with {file} for testing'):
                for image_path, annotation in get_face(file):
                    image = cv2.imread(image_path)
                    for offset in offsets:
                        output_path = os.path.join('data', 'test', f'neg_{cnt}.jpg')
                        process_annotation(image, output_path, annotation, offset)
                        f.write(f'{output_path} 0\n')
                        cnt += 1
