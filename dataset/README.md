## Datasets

> [download.BaiduYun](https://pan.baidu.com/s/1_gYK6iXQDxJnRwlmiTagFw)

### ~~INRIAPerson~~

> ~~[homepage](http://pascal.inrialpes.fr/data/human/)~~

### Caltech

> [homepage](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

> [convert tool](https://github.com/hizhangp/caltech-pedestrian-converter)

#### Usage:

1. Convert .seq to .jpg and convert .vbb to .json
```
	python convert_caltech.py
```


### ~~DukeMTMC-reID~~

> [github.repo](https://github.com/layumi/DukeMTMC-reID_evaluation)

> [BaiduBaike](https://baike.baidu.com/item/%E8%A1%8C%E4%BA%BA%E9%87%8D%E8%AF%86%E5%88%AB/20815009?fr=aladdin#5)

以人搜人，与物体检测不太符合

### RAP

> 主要问题:

> 图片中只有一个标注，但是图片中有很多目标物体。

其中文件detection_boundingboxes_train_refined中记录了用于训练的bounding box集合，包括bounding box坐标值和对应的图片名。需要注意的是：在当前训练数据中，并未标记所有行人的bounding boxes，一张图片中只标注了其中的1~2张行人区域，所以在采集行人负样本时需要人工排查一下。
大家可使用在别的库上训练好的行人model直接在测试集中跑一遍，之后再用训练数据精调一下模型，再跑一遍，将两次检测结果按照一定格式分别保存为两个文件，格式可参考detection_boundingboxes_train_refined。测试集的ground truth暂不提供，大家提交结果后，我们会计算性能指标公开给大家。