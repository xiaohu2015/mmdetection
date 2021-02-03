import numpy as np
import torch
import glob
import onnxruntime
import cv2
import os

def resize(image,_resize_size_hw):
    h, w = image.shape[:2]
    if w == _resize_size_hw[1] and h == _resize_size_hw[0]:
        scale = 1.0
        pad = (0, 0, 0, 0)
    else:
        if w / _resize_size_hw[1] >= h / _resize_size_hw[0]:
            scale = _resize_size_hw[1] / w
        else:
            scale = _resize_size_hw[0] / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w == _resize_size_hw[1] and new_h == _resize_size_hw[0]:
            pad = (0, 0, 0, 0)
        else:
            pad_w = (_resize_size_hw[1] - new_w) / 2
            pad_h = (_resize_size_hw[0] - new_h) / 2
            pad = (int(pad_h), int(pad_h + .5), int(pad_w), int(pad_w + .5))

    parameter_dict={}
    parameter_dict['scale'] = scale
    parameter_dict['pad_tblr'] = pad
    parameter_dict['scale_offset_hw'] = (0, 0)

    if parameter_dict['scale'] != 1:
        new_w = int(w * parameter_dict['scale'])
        new_h = int(h * parameter_dict['scale'])
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        scale_offset_w = new_w / w - parameter_dict['scale']
        scale_offset_h = new_h / h - parameter_dict['scale']
        parameter_dict['scale_offset_hw'] = (scale_offset_h, scale_offset_w)

    top = parameter_dict['pad_tblr'][0]
    bottom = parameter_dict['pad_tblr'][1]
    left = parameter_dict['pad_tblr'][2]
    right = parameter_dict['pad_tblr'][3]

    pad_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return pad_img,parameter_dict


def _convert_origshape(boxes, parameter_dict):
    top = parameter_dict['pad_tblr'][0]
    left = parameter_dict['pad_tblr'][2]
    scale = parameter_dict['scale']
    boxes[:, 0:3:2] -= left
    boxes[:, 1:4:2] -= top
    boxes[:, :4] /= scale
    return boxes


if __name__ == '__main__':

    root='/home/SENSETIME/huanghaian/dataset/project/images/210202-data'

    img_name_list = glob.glob(root + os.sep + '*')
    img_name_list = list(filter(lambda f: f.find('json') < 0, img_name_list))

    ort_session = onnxruntime.InferenceSession("../../tmp.onnx")
    num_classes=['ok','ng']

    for img_path in img_name_list:
        img=cv2.imread(img_path)
        image_copy=img.copy()
        img,para_dict=resize(img,(320,320))
        img=img/255
        img=np.float32(img)
        img1=np.transpose(img,(2,0,1))[None]
        ort_inputs = {ort_session.get_inputs()[0].name: img1}
        ort_outs = ort_session.run(None, ort_inputs)
        bboxes=ort_outs[0] # nx5
        lables=ort_outs[1] # nx1

        # revert to orig img
        bboxes=_convert_origshape(bboxes,para_dict)

        # for bbox in bboxes:
        #     bbox_f = np.array(bbox[:4], np.int32)
        #     image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), (0,0,255),
        #                                5)
        #
        # cv2.namedWindow('img',0)
        # cv2.imshow('img',image_copy)
        # cv2.waitKey(0)

        #





