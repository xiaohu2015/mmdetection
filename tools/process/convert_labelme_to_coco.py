import os.path as osp

import mmcv
from PIL import Image
from tools.process.coco_create import CocoCreator


def _generate_batch_data(sampler, batch_size=5):
    batch = []
    for idx in sampler:
        batch.append(idx)
        if len(batch) == batch_size:
            yield batch
            batch = []


if __name__ == '__main__':
    root_path = '/home/SENSETIME/huanghaian/dataset/project/images'
    out_dir = '/home/SENSETIME/huanghaian/dataset/project/'
    save_name = 'train.json'
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations'))

    # 手动定义
    categories = [{
        'id': 1,
        'name': 'out-ok',
        'supercategory': 'object',
    }, {
        'id': 2,
        'name': 'out-ng',
        'supercategory': 'object',
    }]

    label_dict = {'out-ok': 0, 'out-ng': 1}
    coco_creater = CocoCreator(
        categories, out_dir=out_dir, save_name=save_name)

    segmentation_id = 1
    paths = mmcv.scandir(root_path, '.json', recursive=True)
    for i, path in enumerate(paths):
        image_id = i + 1  # 手动设置

        img_path = osp.join(root_path, path[:-4] + 'jpg')
        image = Image.open(img_path)
        # 第一步
        coco_creater.create_image_info(image_id, path[:-4] + 'jpg', image.size)

        json_path = osp.join(root_path, path)
        json_data = mmcv.load(json_path)
        shapes = json_data['shapes']

        shapes_o = []
        for shape in shapes:
            if shape['label'].lower() == 'out-ng':
                shapes_o.append(shape)
                shapes_o.append([])
                shapes_o.append([])
                shapes_o.append([])
                shapes_o.append([])
            else:
                shapes_o.append(shape)

        assert len(shapes_o) % 5 == 0

        batch_data = _generate_batch_data(shapes_o, 5)

        rectangles = []
        labels = []
        points = []
        for data in batch_data:
            # x1y1x2y2
            x1, y1 = data[0]['points'][0]
            x2, y2 = data[0]['points'][1]
            w = int(x2 - x1)
            h = int(y2 - y1)
            x1 = int(x1)
            x2 = int(x2)

            # bbox + label
            class_id = categories[label_dict[data[0]['label'].lower()]]['id']
            category_info = {'id': class_id, 'is_crowd': 0}

            # 第二步
            coco_creater.create_annotation_info(
                segmentation_id,
                image_id,
                category_info,
                bounding_box=[x1, y1, w, h])
            segmentation_id += 1

    # 第三步
    coco_creater.dump()
