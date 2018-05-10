import numpy as np
import glob
import os
import cv2
from scipy.ndimage.interpolation import rotate

def erase_marks(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    areas = stats[:,4]
    areas<100000
    for i in np.argwhere(areas<100000):
        img[labels==i] = 0
    return img


def get_maxmin(img):
    # img: crop region must be 0 and else 255
    non_zero = np.argwhere(img == 255)
    x_min = np.min(non_zero[:, 0])
    y_min = np.min(non_zero[:, 1])
    x_max = np.max(non_zero[:, 0])
    y_max = np.max(non_zero[:, 1])
    return x_min, y_min, x_max, y_max


def calc_area(img):
    x_min, y_min, x_max, y_max = get_maxmin(img)
    return (x_max-x_min) * (y_max-y_min)


def rotate_min(img):
    # rotate img so that crop region is minimum
    area = calc_area(img)
    angle = 0.0
    num_try = 0
    while True:
        angle += 0.2
        img = rotate(img, angle)
        new_area = calc_area(img)
        if new_area > area:
            num_try += 1
            if num_try == 3:
                break
        else:
            num_try = 0
            area = new_area
    if angle == 0.2:
        num_try = 0
        angle = 0.0
        while True:
            angle -= 0.2
            img = rotate(img, angle)
            new_area = calc_area(img)
            if new_area > area:
                if num_try == 3:
                    break
            else:
                num_try = 0
                area = new_area
    if angle == -0.2:
        angle = 0.0
    return angle


data_path = '../data/'
img_file = glob.glob(os.path.join(data_path, 'original_img/*'))

for num, f in enumerate(sorted(img_file)):
    file_name = f.split('/')[-1]
    print('\rProcessing {}/{}'.format(num+1, len(img_file)), end='')
    # load image
    img = cv2.imread(f)
    img = img[0:2000, :2000]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert to binarize (0, 255)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # fill in small dots
    close_kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=10)

    # fill in card mark and number
    connected = erase_marks(closing)

    # dilate for card edges
    dilate_kernel = np.ones((3, 3), np.uint8)
    dilation = 255 - cv2.dilate(connected, dilate_kernel, iterations=2)

    # rotate image
    angle = rotate_min(dilation)
    new_img = np.c_[img, np.expand_dims(dilation, axis=-1)]
    new_img = rotate(new_img, angle)
    
    # crop
    x_min, y_min, x_max, y_max = get_maxmin(rotate(dilation, angle))
    new_img = new_img[x_min-1:x_max+1, y_min-1:y_max+1]

    # save
    save_dir = os.path.join(data_path, 'cards_data', file_name.split('.')[0]+'.png')
    cv2.imwrite(save_dir, new_img)
