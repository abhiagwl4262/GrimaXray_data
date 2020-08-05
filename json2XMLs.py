import json 
import sys
import dicttoxml
from collections import OrderedDict 
import os
import cv2


## here x means x-axis means column 
## here y means y-axis means row

verify =  False

f_dims_json = open('images_dims.json', 'r')
dims 		= json.load(f_dims_json)

f_json = open('all_data.json', 'r')
anns = json.load(f_json)

for key, val in anns.items():
    image_dict  = OrderedDict()
    im_dict     = OrderedDict()
    im_dict['folder'] = "VOC2007"
    im_dict['filename'] = key + ".png"
    
    im_source = OrderedDict()
    im_source['database']   = "The VOC2007 Database"
    im_source['annotation'] = "PASCAL VOC2007"
    im_source['image']      = "flickr"
    im_source['flickrid']   = 331667550 
    im_dict['source']       = im_source

    im_owner = OrderedDict()
    im_owner['flickrid']= 'Baliwag boy'
    im_owner['name']    = 'jojo puno'
    im_dict['owner']    = im_owner 

    size = OrderedDict()
    size['width'] = 768 #dims[key][1]
    size['height']= 572 #dims[key][0]
    size['depth'] = 3
    im_dict['size'] = size 

    im_dict['segmented'] = 0

    if verify:
    	im = cv2.imread("./VOCdevkit2007/VOC2007/JPEGImages/" + key + ".png")

    for box in val:
        object_dict = OrderedDict()

        object_dict['name']     = 'defect'
        object_dict['pose']     = 'Unspecified'
        object_dict["truncated"]= 0
        object_dict['difficult']= 0

        bbox_dict = OrderedDict()
        bbox_dict['ymin']        = box[0]
        bbox_dict['xmin']        = box[1]
        bbox_dict['ymax']        = box[2]
        bbox_dict['xmax']        = box[3]
        object_dict['bndbox'] = bbox_dict

        if verify:
        	cv2.rectangle(im, (bbox_dict['xmin'], bbox_dict["ymin"]), (bbox_dict['xmax'], bbox_dict["ymax"]), color = (0,255,0), thickness=2)
        
        im_dict["object"] = object_dict

    if verify:
	    cv2.imshow("window", im)
	    cv2.waitKey(0)

    image_dict['annotation'] = im_dict  
    xml = dicttoxml.dicttoxml(image_dict, attr_type=False, root=False)
    wfile = open(os.path.join("./VOCdevkit2007/VOC2007/Annotations",key + ".xml"), 'wb')
    wfile.write(xml)
    wfile.close()
