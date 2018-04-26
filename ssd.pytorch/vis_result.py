import sys
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

import ipdb
from ssd import build_ssd
from data.caltech import CALTECHDetection, CAL_ROOT
from data.rap import RAPDetection, RAP_ROOT
from data import BaseTransform

import platform

print('python version: {}'.format(platform.python_version()))
print('PyTorch version: {}'.format(torch.__version__))

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_CAL_5000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--dataset_root', default=CAL_ROOT)
parser.add_argument('--save_folder', default='vis_dir/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print('GPU ID: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))

torch.set_default_tensor_type('torch.cuda.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def vis_result():
    net = build_ssd(phase='test', size=300, num_classes=2)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = CALTECHDetection(root=args.dataset_root)
    net.cuda()
    cudnn.benchmark = True

    num_images = len(dataset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        img = dataset.pull_image(i)
        # img_id, annotation = dataset.pull_anno(i)
        transform = BaseTransform(net.size, (104, 117, 123))  # resize to 300x300 and subtract channel-wise mean.
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        x = x.cuda()
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        pred_num = 0
        ipdb.set_trace()
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                # if pred_num == 0:

                # score = detections[0, i, j, 0]
                # label_name = labelmap[i - 1]
                # pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                # coords = (pt[0], pt[1], pt[2], pt[3])
                # pred_num += 1
                j += 1


if __name__ == '__main__':
    vis_result()
