import cv2
import os
import json

image_dir = './VOCdevkit2007/VOC2007/JPEGImages'

dict = {}
for im in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,im))
    height = img.shape[0]
    width  = img.shape[1]
    dict[im.split('.')[0]] = [height,width]

f_write = open("images_dims.json", 'w')
json.dump(dict,f_write)
