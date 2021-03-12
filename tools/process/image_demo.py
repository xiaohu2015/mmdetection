from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import glob
import os
import cv2
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='../../configs/yolo/tinyyolov4_coco.py', help='Config file')
    parser.add_argument('--checkpoint', default='../../tools/work_dirs/tinyyolov4_coco/epoch_198.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    args = parser.parse_args()

    root = '/home/SENSETIME/huanghaian/dataset/project/test_img'
    img_name_list = glob.glob(root + os.sep + '*')
    img_name_list = list(filter(lambda f: f.find('json') < 0, img_name_list))

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for img_path in img_name_list:
        print(img_path)
        # if img_path.find('ok/2021_03_06_11_31_53-1')<0:
        #     continue
        img = cv2.imread(img_path)
        # test a single image
        result = inference_detector(model, img)

        # print('ng num=', sum(result[1]), 'total=', len(result[1]))
        # for i, (bbox, label) in enumerate(zip(result[0], result[1])):
        #     bbox_f = np.array(bbox[:4].cpu(), np.int32)
        #     if label.cpu() == 0:
        #         image_copy = cv2.rectangle(img, (bbox_f[0], bbox_f[1]),
        #                                    (bbox_f[2], bbox_f[3]), (255, 0, 0),
        #                                    5)
        #     else:
        #         image_copy = cv2.rectangle(img, (bbox_f[0], bbox_f[1]),
        #                                    (bbox_f[2], bbox_f[3]), (0, 0, 255),
        #                                    5)
        #
        # cv2.namedWindow('img', 0)
        # cv2.imshow('img', image_copy)
        # cv2.waitKey(0)

        print('ng num=', len(result[1]), 'total=', sum([len(result[0]), len(result[1])]))
        show_result_pyplot(model, img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
