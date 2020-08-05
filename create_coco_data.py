import json 
import sys
from collections import OrderedDict 
import os
import shutil
import cv2

verify = False

root_dir = "/mnt/data/abhishek/data/GrimaXray_data"

voc_image_dir  = os.path.join(root_dir, "VOCdevkit2007/VOC2007/JPEGImages/")
coco_image_dir = os.path.join(root_dir, "coco/images/")
coco_label_dir = os.path.join(root_dir, "coco/labels/")

if not os.path.exists(os.path.join(coco_label_dir, "val2014")):
    os.makedirs(os.path.join(coco_label_dir, "val2014"))
if not os.path.exists(os.path.join(coco_label_dir, "train2014")):
    os.makedirs(os.path.join(coco_label_dir, "train2014"))

if not os.path.exists(os.path.join(coco_image_dir, "val2014")):
    os.makedirs(os.path.join(coco_image_dir, "val2014"))
if not os.path.exists(os.path.join(coco_image_dir, "train2014")):
    os.makedirs(os.path.join(coco_image_dir, "train2014"))

train_list  = open("VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt", "r").readlines() 
test_list   = open("VOCdevkit2007/VOC2007/ImageSets/Main/test.txt", "r").readlines() 

train_f  = open("coco/train.txt", "w+") 
test_f   = open("coco/test.txt", "w+") 

images_dict = {}
images_dict['train'] = [im.strip("\n") for im in train_list]
images_dict['test'] = [im.strip("\n") for im in test_list]

f_dims_json = open('images_dims.json', 'r')
dims        = json.load(f_dims_json)

f_json = open('all_data.json', 'r')
anns = json.load(f_json)

for key, val in anns.items():

    width = 768 #dims[key][1]
    height= 572 #dims[key][0]
    depth = 3


    if key in images_dict['test']:
        wfile = open(os.path.join(coco_label_dir, "val2014", key + ".txt"), 'w+')
        shutil.copy(voc_image_dir+key+".png",os.path.join(coco_image_dir, "val2014" , key + ".png"))
        test_f.write(os.path.join(coco_image_dir, "val2014" , key + ".png\n"))
    else:
        wfile = open(os.path.join(coco_label_dir, "train2014", key + ".txt"), 'w+')
        shutil.copy(voc_image_dir+key+".png",os.path.join(coco_image_dir, "train2014", key + ".png"))
        train_f.write(os.path.join(coco_image_dir, "train2014" , key + ".png\n"))

    if verify:
        im = cv2.imread(voc_image_dir+key+".png")

    for box in val:
        object_dict = OrderedDict()

        class_idx   = 0
        # xmin        = box[0]
        # ymin        = box[1]
        # xmax        = box[2]
        # ymax        = box[3]  
        
        ymin        = box[0]        
        xmin        = box[1]
        ymax        = box[2]
        xmax        = box[3]  
        

        if verify:
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color = (255, 0, 0), thickness = 2)


        box_width   =  (xmax - xmin)     
        box_height  =  (ymax - ymin)       
        x_center    =  (xmin + box_width/2)       
        y_center    =  (ymin + box_height/2)

        # box_width   =  (ymax - ymin)     
        # box_height  =  (xmax - xmin)       
        # x_center    =  (xmin + box_height/2)       
        # y_center    =  (ymin + box_width/2)

        box_width = box_width/float(width)
        box_height= box_height/float(height)
        x_center  = x_center/float(width)
        y_center  = y_center/float(height)
        
        # box_width = box_width/float(height)
        # box_height= box_height/float(width)        
        # x_center  = x_center/float(height)
        # y_center  = y_center/float(width)

        if box_width > 1.0 or box_height > 1.0 or x_center > 1.0 or y_center > 1.0:
            print(key)
            cv2.imwrite(key+"_det.png", im)
        ann_line = str(class_idx) + " " + str(x_center) + " " + str(y_center) + " " + str(box_width) + " " + str(box_height) + "\n"       
        wfile.write(ann_line)

    if verify:
        cv2.imshow("window", im)
        cv2.waitKey(0)
    wfile.close()
