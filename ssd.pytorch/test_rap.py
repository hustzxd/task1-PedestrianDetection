import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
import csv
from ssd import build_ssd

# from matplotlib import pyplot as plt
from data import COCOPersonDetection, VOCPersonDetection, RAPDetection


import ipdb
import platform

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_VOC_person_AP@77.80_115000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--threshold', default=0.05, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


print('python version: {}'.format(platform.python_version()))
print('PyTorch version: {}'.format(torch.__version__))
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print('GPU ID: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def saveCSV(filename, data):
    assert filename is not None
    assert data is not None
    is_csv = filename.find('.csv')
    assert is_csv > 0

    titles = []
    for key, value in data[0].items():
        titles.append(key)
    with open(filename, 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=titles)
        csv_writer.writeheader()
        csv_writer.writerows(data)
    return

def test_rap():
    model = build_ssd('test', 300, 2)    # initialize SSD
    model.load_weights(args.trained_model)
    testset = RAPDetection(phase='test') # should be 'test'
    result = list()
    index = 0
    start = time.time()
    for i in range(len(testset)):
        # if i > 100:
            # break
        if i % 20 == 0:
            print('Testing image {}/{}'.format(i+1, len(testset)))
        image, filename = testset.pull_image(i)
        # print(image.shape) # (357, 500, 3)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(image, (300, 300)).astype(np.float32) # int to float32
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1) #HWC => CHW
        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = model(xx)

        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= args.threshold:
                # ipdb.set_trace()
                score = detections[0,i,j,0].cpu().item()
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                # coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                # color = colors[i]   
                # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                j+=1
                # ipdb.set_trace()
                res = {}
                res['index'] = index
                res['x'] = '{:.2f}'.format(pt[0])
                res['y'] = '{:.2f}'.format(pt[1])
                res['width'] = '{:.2f}'.format(pt[2] - pt[0] + 1)
                res['height'] = '{:.2f}'.format(pt[3] - pt[1] + 1)
                res['score'] = '{:.2f}'.format(score)
                res['filename'] = filename
                result.append(res)
                index += 1
    res_filename = os.path.join(args.save_folder, '{}.csv'.format(args.trained_model.split('/')[-1]))
    saveCSV(res_filename, result)
    end = time.time()
    print('result has been saved in {}'.format(res_filename))
    print('Total Time: {}s'.format(end-start))
    print('FPS: {}'.format(len(testset) / (end-start)))
if __name__ == '__main__':
    test_rap()