import os
import json
import pickle
import pathlib
import argparse
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
from PIL import ImageFilter

import random
from collections import defaultdict
from torch.utils.data import ConcatDataset
from utils.dataset import CUBWithFilename, PetWithFilename, Flowers102WithFilename

import numpy as np
import cv2
from torchvision import transforms


#mask aug
def blur_boxes(image, boxes, radius=10):
    """
    bbox領域だけGaussian Blurを適用する

    :param image: PIL Image
    :param boxes: [[x_min, y_min, x_max, y_max], ...]
    :param radius: blur強さ
    :return: PIL Image
    """

    img = image.copy()

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)

        # bbox部分を切り出し
        region = img.crop((x_min, y_min, x_max, y_max))

        # Gaussian Blur
        region = region.filter(ImageFilter.GaussianBlur(radius=radius))

        # 元画像に戻す
        img.paste(region, (x_min, y_min))

    return img


def mask_boxes(image, boxes, labels=None, mask_color=(0, 0, 0)):
    """
    画像の指定bbox領域をマスクする。

    :param image: PIL Image
    :param boxes: [[x_min, y_min, x_max, y_max], ...]
    :param labels: (任意) ラベルリスト
    :param mask_color: マスク色 (default: black)
    :return: masked PIL Image
    """

    draw = ImageDraw.Draw(image)

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)

        # bbox領域を塗りつぶし
        draw.rectangle([x_min, y_min, x_max, y_max], fill=mask_color)

    return image

# def mask_boxes(image, boxes):
#     """
#     bbox領域をランダムノイズでマスクする
#     """

#     img = np.array(image)

#     for box in boxes:
#         x_min, y_min, x_max, y_max = map(int, box)

#         h = y_max - y_min
#         w = x_max - x_min

#         # ランダムノイズ生成
#         noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

#         img[y_min:y_max, x_min:x_max] = noise

#     return Image.fromarray(img)



def random_mask(image, boxes):
    img = image.copy()
    fill_color = (0,0,0)
    img = mask_boxes(img, boxes, mask_color=fill_color)
    # img = mask_boxes(img, boxes)
    # img = blur_boxes(img, boxes)
    aug_info = {
        "type": "mask",
        "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
        "color": fill_color
    }
    return img, aug_info



### crop aug
def bbox_overlap_ratio(box, crop):

    xA = max(box[0], crop[0])
    yA = max(box[1], crop[1])
    xB = min(box[2], crop[2])
    yB = min(box[3], crop[3])

    inter_w = max(0, xB-xA)
    inter_h = max(0, yB-yA)

    inter = inter_w * inter_h

    area_box = (box[2]-box[0])*(box[3]-box[1])

    if area_box == 0:
        return 0

    return inter / area_box

def iou_part_crop(
    img,
    boxes,
    labels,
    crop_ratio=(0.3,0.6),
    min_visible_parts=1,
    vis_thresh=0.3,
    max_trial=10
):
    """
    IoUベース part-aware crop

    Args
        img: PIL image
        boxes: [[x1,y1,x2,y2], ...]
        labels: ["head","wing"...]
        crop_ratio: cropサイズ
        min_visible_parts: 最低可視パーツ
        iou_thresh: partが見えていると判定するIoU
    """

    W,H = img.size

    for _ in range(max_trial):

        scale = random.uniform(*crop_ratio)

        new_W = int(W*scale)
        new_H = int(H*scale)

        left = random.randint(0, W-new_W)
        top = random.randint(0, H-new_H)

        right = left + new_W
        bottom = top + new_H

        crop_box = [left,top,right,bottom]

        visible_boxes = []
        visible_labels = []

        for box,label in zip(boxes,labels):

            # iou = compute_iou(box,crop_box)
            # iou = compute_iou(crop_box, box)
            ratio = bbox_overlap_ratio(box, crop_box)

            if ratio > vis_thresh:

                new_box = [
                    box[0]-left,
                    box[1]-top,
                    box[2]-left,
                    box[3]-top
                ]

                #TypeError: Object of type float32 is not JSON serializable
                visible_boxes.append([float(x) for x in new_box])
                visible_labels.append(label)


        if len(labels) > len(visible_boxes) >= min_visible_parts:

            cropped = img.crop((left,top,right,bottom))
            aug_info = {
                "type": "crop",
                "crop_box": [left,top,right,bottom],
                "visible_boxes": visible_boxes,
                "visible_labels": visible_labels
            }

            return cropped, visible_boxes, visible_labels, aug_info

    return img, boxes, labels, {}



def hue_shift_boxes(image, boxes):

    img_np = np.array(image)

    hue_shift = random.randint(-10,10)
    sat_scale = random.uniform(0.8,1.2)

    for box in boxes:

        x1,y1,x2,y2 = map(int,box)

        region = img_np[y1:y2, x1:x2]

        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)

        # 型変換して安全に計算
        h = hsv[:,:,0].astype(np.int16)
        h = (h + hue_shift) % 180
        hsv[:,:,0] = h.astype(np.uint8)

        s = hsv[:,:,1].astype(np.float32)
        s = np.clip(s * sat_scale, 0, 255)
        hsv[:,:,1] = s.astype(np.uint8)

        region = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        img_np[y1:y2, x1:x2] = region
    aug_info = {
        "type": "color",
        "boxes": boxes.tolist() if hasattr(boxes, "tolist") else boxes,
        "hue_shift": hue_shift,
        "sat_scale": sat_scale
    }

    return Image.fromarray(img_np), aug_info



def recolor_bbox_center_color(image, boxes):

    img = np.array(image)

    for box in boxes:

        x1,y1,x2,y2 = map(int,box)

        region = img[y1:y2, x1:x2]

        h, w = region.shape[:2]

        cx = w // 2
        cy = h // 2

        sample = region[
            max(0,cy-5):min(h,cy+5),
            max(0,cx-5):min(w,cx+5)
        ]

        center_color = sample.reshape(-1,3).mean(axis=0)

        dist = np.linalg.norm(region - center_color, axis=2)
        mask = dist < 40

        # texture
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # --- 黒対策 ---
        gray = np.maximum(gray, 0.3)

        # random color
        color = np.random.randint(0,256,3)

        recolor = (gray[...,None] * color).astype(np.uint8)

        region[mask] = recolor[mask]

        img[y1:y2, x1:x2] = region

    aug_info = {
        "type": "color",
        "boxes": boxes.tolist() if hasattr(boxes, "tolist") else boxes,
    }

    return Image.fromarray(img), aug_info

def recolor_bbox_lab(image, boxes):

    img = np.array(image)

    for box in boxes:

        x1,y1,x2,y2 = map(int,box)

        region = img[y1:y2, x1:x2]

        lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)

        L = lab[:,:,0]

        # ランダム色
        a = random.randint(50,200)
        b = random.randint(50,200)

        lab[:,:,1] = a
        lab[:,:,2] = b

        region = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        img[y1:y2,x1:x2] = region
    aug_info = {
        "type": "color",
        "boxes": boxes.tolist() if hasattr(boxes, "tolist") else boxes,
    }

    return Image.fromarray(img), aug_info


def recolor_image_lab(image):

    img = np.array(image)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    lab[:,:,1] = random.randint(50,200)
    lab[:,:,2] = random.randint(50,200)

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img)




def augmentation(image, results, N=3, ops=["crop","mask"], image_recolor=False, recolor_ratio=0.3):

    labels = results['labels']
    grouped = defaultdict(list)
    for i, l in enumerate(labels):
        grouped[l].append(i)

    outputs = []
    for _ in range(N):
        selected = random.choice(ops)
        if len(grouped) == 0:
            selected = "crop"
        if selected == "mask":
            key = random.choice(list(grouped.keys()))
            value = grouped.pop(key)
            img, info = random_mask(image.copy(), results["boxes"][value].numpy())
            info['concpet'] = key

        elif selected == "crop":
            img, new_boxes, new_labels, info = iou_part_crop(
                image.copy(),
                results["boxes"].numpy(),
                results["labels"],
                min_visible_parts=1,
                vis_thresh=0.4
            )
        elif selected == "color":
            #あまりうまくいかない
            key = random.choice(list(grouped.keys()))
            value = grouped.pop(key)
            # img, info = hue_shift_boxes(image.copy(), results["boxes"][value])
            img, info = recolor_bbox_lab(image.copy(), results["boxes"][value])
            info['concept'] = key
        
        if image_recolor:
            if random.random() <= recolor_ratio:
                # img = recolor_image_lab(img)
                #color_jitterが一番自然
                img = transforms.ColorJitter(
                    saturation=2.0,
                    hue=0.3
                )(img)

                info['recolor'] = True
            else:
                info['recolor'] = False

        outputs.append((img, info))

    return outputs

                

def main(args):
    
    with open(args.llmdet_output_path, 'rb') as f:
        llmdet_outputs = pickle.load(f)

    if args.dataset == 'cub':
        dataset = CUBWithFilename(root=args.data_dir + '/cub/CUB_200_2011/images/train', transform=None, mode='all')
        image_recolor=False
    elif args.dataset == 'pet':
        dataset = PetWithFilename(root=args.data_dir + '/oxford_pet', split="trainval", transform=None, mode='all')
        # dataset = PetWithFilename(root=args.data_dir + '/oxford_pet', split="test", transform=None, mode='all')
        class_to_idx = dataset.class_to_idx   # 例: {"Abyssinian": 0, "American_Bulldog": 1 ...}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        image_recolor=True
    elif args.dataset == 'flower':
        trainset = Flowers102WithFilename(root=args.data_dir, transform=None, split='train', mode='all')
        valset = Flowers102WithFilename(root=args.data_dir, transform=None, split='val', mode='all')
        dataset = ConcatDataset([trainset, valset])
        image_recolor=True
        with open('../descriptors/my_flowers.json', 'r') as f:
            descriptors = json.load(f)
        classes = trainset.classes
    
    metadata = {}
    for sample in tqdm(dataset):
        image = sample[0]
        label_id = sample[1]
        filename = sample[2]
        # full_path = sample[-1]

        results = llmdet_outputs[str(filename).lower()]
        output_images = augmentation(image, results, ops=["crop", "mask"], image_recolor=image_recolor, recolor_ratio=args.recolor_ratio)
        # output_images = augmentation(image, results, ops=["crop", "mask", "color"])
        # output_images = augmentation(image, results, ops=["color"])

        os.makedirs(f'{args.output_dir}/{label_id}', exist_ok=True)
        for i in range(args.N):
            img, info = output_images[i]
            out_filename = f"{filename.replace('.jpg','')}_{i}.jpg"
            save_path = f"{args.output_dir}/{label_id}/{out_filename}"
            img.save(save_path)
            metadata[f"{label_id}/{out_filename}"] = info

    meta_path = os.path.join(args.output_dir, "augmentation_meta.json")
    with open(meta_path,"w") as f:
        json.dump(metadata,f,indent=2)
    print(f'saved {args.output_dir}/*')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/user/Develop/datasets')
    parser.add_argument('--dataset', type=str, default='cub')
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--llmdet_output_path', type=str, default='output_llmdet_cub_final.pkl')
    parser.add_argument('--output_dir', type=str, default='/home/user/Develop/datasets/cub/CUB_200_2011/images/train_aug')
    parser.add_argument('--recolor_ratio', type=float, default=0.3)

    args = parser.parse_args()

    main(args)