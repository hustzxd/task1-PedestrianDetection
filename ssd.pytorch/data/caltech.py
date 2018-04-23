"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot

Updated by hustzxd for custom data(Caltech) April 22, 2018.
"""
# from .config import HOME
import os
import os.path as osp
import json
import torch
import torch.utils.data as data
import cv2
import numpy as np
import ipdb

CAL_CLASSES = (  # always index 0
    'person', 'people', 'person-fa', 'person?')

HOME = os.path.expanduser("~")
# note: if you used our download scripts, this should be right
CAL_ROOT = osp.join(HOME, "datasets/Caltech/data")


class CaltechAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(CAL_CLASSES, range(len(CAL_CLASSES))))

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for anno in target:
            pos = anno['pos']  # [xmin, ymin, w, h]
            pos = [float(i) for i in pos]
            bndbox = [pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]  # [xmin, ymin, xmax, ymax]
            lbl = anno['lbl']
            # print(lbl)
            label_idx = 0
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class CALTECHDetection(data.Dataset):
    """Caltech Detection Dataset Object

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
            (default: 'CALTECH')
    """

    def __init__(self, root, image_dir=None, anno_file=None,
                 transform=None, target_transform=CaltechAnnotationTransform(), dataset_name='CALTECH'):
        if image_dir is None:
            image_dir = 'test_images'
        if anno_file is None:
            anno_file = 'test_annotations.json'
        self.root = root
        self.image_dir = image_dir
        self.anno_file = anno_file
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self._base_image_path = os.path.join(root, image_dir, '{}')

        # for image in os.listdir(os.path.join(root, image_dir)):
        #     self.ids.append(image)
        _base_image_id = 'img{:02}{:02}{:04}.jpg'
        self.anno_json = self.load_json()
        for set_key in self.anno_json:
            for num_key in self.anno_json[set_key]:
                for frame_key in self.anno_json[set_key][num_key]['frames']:
                    img_id = _base_image_id.format(int(set_key), int(num_key), int(frame_key))
                    self.ids.append(img_id)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def load_json(self):
        anno_path = os.path.join(self.root, self.anno_file)
        with open(anno_path, 'r') as rf:
            data = json.load(rf)
        return data

    def pull_item(self, index):
        img_id = self.ids[index]
        # print('img_id:{}'.format(img_id))
        # print(self._base_image_path.format(img_id))
        img = cv2.imread(self._base_image_path.format(img_id))
        height, width, channels = img.shape
        # ipdb.set_trace()
        target = self.anno_json[img_id[3:5]][img_id[5:7]]['frames']
        frame_key = img_id[7:11]
        frame_key = str(int(frame_key))
        if frame_key in target:
            target = target[frame_key]
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            raise Exception('anno not found.')
        if self.transform is not None and target is not None:
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
    caltech_dataload = CALTECHDetection(root=CAL_ROOT)
    print(caltech_dataload.__len__())
    caltech_dataload.pull_item(0)
