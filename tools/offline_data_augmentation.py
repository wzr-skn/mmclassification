# from torchvision.transforms import transforms
import os
import cv2
from argparse import ArgumentParser
import warnings
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
from PIL import Image

# ia.seed(1)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image path')
    parser.add_argument('out_file', help='Output file')
    args = parser.parse_args()
    return args


def main(args):

    seq = iaa.SomeOf((1, 4),
          [ iaa.Fliplr(1),
            # change brightness, doesn't affect BBs
            iaa.Multiply((0.5, 1.5)),
            # 从最邻近像素中取均值来扰动
            iaa.AverageBlur((3,3)),
            iaa.MedianBlur(),
            iaa.Sharpen(),
            iaa.Emboss(),
            iaa.AdditiveGaussianNoise(),
            # iaa.GaussianBlur(sigma=(0, 2.0)),
            iaa.Dropout((0, 0.3)),
            iaa.ContrastNormalization(),
            iaa.Affine(rotate=(-5, 5)),
            iaa.Affine(scale=(0.8, 1.0)),
            iaa.Affine(shear=(-5, 5)),
            iaa.PiecewiseAffine(),
            iaa.ElasticTransformation()
    ])


    test_file_list = os.listdir(args.img_path)
    for img_name in test_file_list:
        if img_name[-3:] not in ["jpg", "png", "bmp"] and img_name[-4:] not in ["jpeg"]:
            warnings.warn(f"{img_name} is not a image name")
            continue
        split_name = img_name[:].replace(' ','')
        # split_name =img_name[:]
        image_path = os.path.join(args.img_path, split_name)
        if img_name[-3:] in ["jpg", "png", "bmp"]:
            output_path = os.path.join(args.out_file, split_name[:-4] + "_aug" + split_name[-4:])
        else:
            output_path = os.path.join(args.out_file, split_name[:-5] + "_aug" + split_name[-5:])
        image = cv2.imread(image_path)
        if image is None:
            print(image_path)
        crop_image = image[:, :, :]
        images_aug = seq.augment_image(crop_image)
        cv2.imwrite(output_path, images_aug)



if __name__ == '__main__':
    args = parse_args()
    main(args)
