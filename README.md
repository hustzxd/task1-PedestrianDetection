## Pedestrian detection

> Video processing analysis assignment.

> SSD for pedestrian detection in RAP dataset.

### Train
- First download the fc-reduced VGG-16 PyTorch base network weights at: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

- By default, we assume you have downloaded the file in the ssd.pytorch/weights dir:

```bash
cd ssd.pytorch
# train SSD300 on MS COCO person dataset
./train_coco_person.sh
# train SSD300 on Pascal VOC 2007+2012 dataset
./train_voc_person.sh
# train SSD300 on both COCO and VOC dataset
./train_voc_coco_perosn.sh
```

### Evaluation

```bash
cd ssh.pytorch
# eval on Pascal VOC 2007
./eval_voc_person.sh weights/<you.pth>

# eval on RAP test dataset
python test_rap.py --trained_model weight/<you.pth>
```



### Performace

We just evaluation AP of person in Pascal VOC 2007.

| Training Data| Original | Only person(this project) |
|:-:|:-:|:-:|
| 07+12 | 76.2 % | 77.8% |
| 07+12+COCO | 81.4% | running |

## Reference
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- Thanks to [amdegroot](https://github.com/amdegroot/ssd.pytorch).
