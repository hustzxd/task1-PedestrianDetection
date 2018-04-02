import json
# import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg
import sys
import os

import pdb


class Bbox(object):
	def __init__(self, x1, y1, w, h, category=-1):
		self.x1 = x1
		self.y1 = y1
		self.w = w
		self.h = h
		self.category = category

	def __str__(self):
		return '({x1} {y1} {w} {h} {category})'.format(self.x1, self.y1, self.w, self.h, self.category)


'''
data['frames'][10][10][]
'''
def loadjson(path):
	with open(path, 'r') as rf:
		data = json.load(rf)
		return data


def draw(img_name, bboxes, dst_img_name):
	im = Image.open(img_name)
	draw = ImageDraw.Draw(im)
	for bbox in bboxes:
		left = bbox.x1
		top = bbox.y1
		right = bbox.x1 + bbox.w
		bot = bbox.y1 + bbox.h
		color = (255, 0, 0)
		draw.line((left, top, right, top), fill=color, width=2)
		draw.line((right, top, right, bot), fill=color, width=2)
		draw.line((left, top, left, bot), fill=color, width=2)
		draw.line((left, bot, right, bot), fill=color, width=2)
	# im.show()
	im.save(dst_img_name)
	# pdb.set_trace()


def main():
	img_path_base = os.path.join(images_dir, 'img1000{:04}.jpg')
	img_dst_path_base = os.path.join(dst_images_dir, 'img1000{:04}.jpg')
	anno_json = loadjson(ann_path)
	anno_json = anno_json['10']['00']['frames']
	for frame_key in anno_json.iterkeys():
		frame_key = int(frame_key)
		img_path = img_path_base.format(frame_key)
		img_dst_path = img_dst_path_base.format(frame_key)
		print(img_path)
		# pdb.set_trace()
		annos = anno_json[str(frame_key)]
		bboxes = []
		for anno in annos:
			pos = anno['pos']
			print('pos: {}'.format(pos))
			pos = [float(i) for i in pos]
			lbl = anno['lbl']
			print('lbl: {}'.format(lbl))
			bbox = Bbox(pos[0], pos[1], pos[2], pos[3], category=lbl)
			bboxes.append(bbox)
		draw(img_path, bboxes, img_dst_path)


if __name__ == '__main__':
	ann_path = 'test_annotations.json'
	images_dir = 'mini_images'
	dst_images_dir = 'mini_image_anno'
	if not os.path.exists(dst_images_dir):
		print('mkdir')
		os.mkdir(dst_images_dir)
	main()