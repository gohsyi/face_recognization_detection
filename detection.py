import os, random
import cv2
import numpy as np

from common.argparser import args


def detect_face(model, img):
    """
    Use sliding window to detect faces in img with model
    """

    img = img.copy()
    img_h, img_w, img_c = img.shape
    stride_w, stride_h = (args.stride, args.stride)
    res = []

    for scale in args.scales:
        for hw_ratio in args.hw_ratios:
            winSize = args.winSize * scale
            bbox_w, bbox_h = int(winSize * hw_ratio), int(winSize)

            for c in range((img_w - bbox_w) // stride_w + 1):
                for r in range((img_h - bbox_h) // stride_h + 1):
                    tl_x, tl_y = c * stride_w, r * stride_h  # top-left point
                    bbox = np.array([tl_x, tl_x+bbox_w, tl_y, tl_y+bbox_h], dtype=int)
                    score = model.score(img[bbox[2]:bbox[3], bbox[0]:bbox[1], :])

                    if score > args.thres_score:
                        overlap = []
                        for i in range(len(res)):
                            iou = IoU(bbox, res[i][1])
                            if iou > args.thres_iou:
                                if score > res[i][0]:
                                    overlap.append(i)
                                else:
                                    break
                        else:
                            res.append((score, bbox))

                        # trick, pop index
                        for i in range(len(overlap)):
                            res.pop(overlap[i] - i)

    # Draw Bounding Boxes
    for _, bbox in res:
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.rectangle(img, (bbox[0],bbox[2]), (bbox[1],bbox[3]), color, 2)

    return res, img


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


if not os.path.exists('logs/detection/'):
    os.makedirs('logs/detection/')

from recognization import model

for root, dirs, files in os.walk('data/originalPics/'):
    for img_file in files:
        if os.path.splitext(img_file)[-1] == '.jpg':
            img_path = os.path.join(root, img_file)
            img = cv2.imread(img_path)
            res, img = detect_face(model, img)

            cv2.imwrite(f'logs/detection/{img_file}', img)
            print(res)
