import asyncio
from argparse import ArgumentParser
import os
import warnings


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_file', help='Image file')
    parser.add_argument('out_file', help='output file')
    args = parser.parse_args()
    return args


def main(args):
    img_file_list = os.listdir(args.img_file)
    number = 1
    annotations_train = []
    name = "hands_off"
    for img_name in img_file_list:
        if  img_name[-3:] not in ["jpg", "png", "bmp"] and img_name[-4:] not in ["jpeg"]:
            warnings.warn(f"{img_name} is not a image name")
            continue
        if img_name[-4] == ".":
            img_path = os.path.join(args.img_file, img_name)
            out_path = os.path.join(args.out_file,  name + "0" + format(str(number), "0>3s") + img_name[-4:])
            filename_gt = name + "0" + format(str(number), "0>3s") + img_name[-4:] + " " + "0"
            os.rename(img_path, out_path)
            number += 1
            annotations_train.append(filename_gt)
        else:
            img_path = os.path.join(args.img_file, img_name)
            out_path = os.path.join(args.out_file,  name + "0" + format(str(number), "0>3s") + img_name[-5:])
            filename_gt = name + "0" + format(str(number), "0>3s") + img_name[-5:] + " " + "0"
            os.rename(img_path, out_path)
            number += 1
            annotations_train.append(filename_gt)

    with open("../../../../media/traindata/coco_crop_data/val_crop_hands_state_filted/hands_off/hands_off.txt", "a+") as ff:
        for i in range(len(annotations_train)):
            ff.write(str(annotations_train[i])+"\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
