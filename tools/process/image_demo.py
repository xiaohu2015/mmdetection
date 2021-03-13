from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import glob
import os
import cv2
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='../../configs/yolo/tinyyolov4_coco.py', help='Config file')
    parser.add_argument('--checkpoint', default='../../tools/work_dirs/tinyyolov4_coco/latest.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    root = '/home/hha/dataset/project/images/ok'
    img_name_list = glob.glob(root + os.sep + '*')
    img_name_list = list(filter(lambda f: f.find('json') < 0, img_name_list))

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for img_path in img_name_list:
        print(img_path)
        if img_path.find('ok/2021_03_06_11_24_06-1.jpg')<0:
            continue
        img = cv2.imread(img_path)
        # test a single image
        bbox_result = inference_detector(model, img)

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        print('ng num=', sum(labels), 'total=', len(bboxes))
        for i, (bbox, label) in enumerate(zip(bboxes,labels)):
            bbox_f = np.array(bbox[:4], np.int32)
            if bbox[4]<args.score_thr:
                continue
            if label == 0:
                image_copy = cv2.rectangle(img, (bbox_f[0], bbox_f[1]),
                                           (bbox_f[2], bbox_f[3]), (255, 0, 0),
                                           5)

            else:
                image_copy = cv2.rectangle(img, (bbox_f[0], bbox_f[1]),
                                           (bbox_f[2], bbox_f[3]), (0, 0, 255),
                                           5)
            cv2.putText(image_copy, str(bbox[4]), (bbox_f[0], bbox_f[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        cv2.namedWindow('img', 0)
        cv2.imshow('img', image_copy)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
