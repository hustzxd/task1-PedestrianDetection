"""RAP Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot

Updated by hustzxd for custom data(RAP) April 23, 2018.
"""
# from .config import HOME
import os
import os.path as osp
import json
import torch
import torch.utils.data as data
import cv2
import numpy as np
import csv
import ipdb

HOME = os.path.expanduser("~")
# note: if you used our download scripts, this should be right
RAP_ROOT = osp.join(HOME, "datasets/RAP-Detection")
"""
➜  RAP-Detection ls
detection_boundingboxes_train_refined.csv  file_operation.py  test_jpg  test_jpg.tar.gz  train_jpg  train_jpg.tar.gz
"""


def loadCSV(filename):
    """
    Load a csv file into the format of a list of dictionary.

    filename - the filename of the csv to be loaded.
    return a list of dictonary.
    """
    is_existed = os.path.exists(filename)
    assert is_existed is True

    data = []
    with open(filename) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        # headers = csv_reader.fieldnames
        for line in csv_reader:
            data.append(line)
        # print csv_reader
    return data


class RAPAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
    """

    def __init__(self):
        pass

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        index = target['index']
        h = float(target['height'])
        w = float(target['width'])
        y = float(target['y'])
        x = float(target['x'])
        bndbox = [x / width, y / height, (x + w) / width, (y + h) / height]  # [xmin, ymin, xmax, ymax]
        label_idx = 0
        bndbox.append(label_idx)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class RAPDetection(data.Dataset):
    """RAP Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to Caltech folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'RAP')
    """

    def __init__(self, root, image_dir=None, anno_file=None,
                 transform=None, target_transform=RAPAnnotationTransform(), dataset_name='RAP'):
        if image_dir is None:
            image_dir = 'train_jpg'
        if anno_file is None:
            anno_file = 'detection_boundingboxes_train_refined.csv'
        self.root = root
        self.image_dir = image_dir
        self.anno_file = anno_file
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.anno_data = loadCSV(os.path.join(root, anno_file))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.anno_data)

    def pull_item(self, index):
        target = self.anno_data[index]
        index = target['index']
        filename = target['filename']
        img = cv2.imread(os.path.join(self.root, self.image_dir, filename))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target,width=width, height=height)
            # print(target)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            # img = img[:, :, (2, 1, 0)]  # why bgr to rgb???
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    # def pull_image(self, index):
    #     '''Returns the original image object at index in PIL form
    #
    #     Note: not using self.__getitem__(), as any transformations passed in
    #     could mess up this functionality.
    #
    #     Argument:
    #         index (int): index of img to show
    #     Return:
    #         PIL img
    #     '''
    #     img_id = self.ids[index]
    #     return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
    #
    # def pull_anno(self, index):
    #     '''Returns the original annotation of image at index
    #
    #     Note: not using self.__getitem__(), as any transformations passed in
    #     could mess up this functionality.
    #
    #     Argument:
    #         index (int): index of img to get annotation of
    #     Return:
    #         list:  [img_id, [(label, bbox coords),...]]
    #             eg: ('001718', [('dog', (96, 13, 438, 332))])
    #     '''
    #     img_id = self.ids[index]
    #     anno = ET.parse(self._annopath % img_id).getroot()
    #     gt = self.target_transform(anno, 1, 1)
    #     return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


if __name__ == '__main__':
    rap_dataset = RAPDetection(root=RAP_ROOT)
    print(rap_dataset.__len__())
    im, gt = rap_dataset.__getitem__(0)
    print(gt)
