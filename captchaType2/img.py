from __future__ import print_function, absolute_import, division
from math import floor, ceil
from skimage import img_as_ubyte
from skimage.measure import find_contours
from skimage.util import crop
from skimage.transform import resize
from skimage.transform import rescale
import numpy as np

from sklearn.cluster import KMeans
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

BINARY_THRESH = 30 # image binary thresh
LETTER_SIZE = (16, 14) # letter heigth,width
FULL_IMG_W = 85

def sortByArea(elem):
    return elem[1]

def sortByIndex(elem):
    return elem[0]

def get_color_str(colorArr):
    return "{},{},{}".format(colorArr[0], colorArr[1],colorArr[2])

def color_quantization(img):
    n_colors = 16
    img = np.array(img, dtype=np.float64) / 255

    w, h, d = original_shape = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    labels = kmeans.predict(image_array)

    codebook = kmeans.cluster_centers_
    image = np.zeros((w, h, codebook.shape[1]))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1

    return image

def contour_filter(img_data,contour,area):
    w = contour[1][0] - contour[0][0] # width
    h = contour[1][1] - contour[0][1] # height

    pix_count = 0

    for i in range(contour[0][1],contour[1][1]):
        for j in range(contour[0][0],contour[1][0]):
            if img_data[i][j] == 0:
                pix_count += 1

    if pix_count/area < 0.15 or w / h > 2:
        for i in range(contour[0][1],contour[1][1]):
            for j in range(contour[0][0],contour[1][0]):
                img_data[i][j] = 255

    return img_data

def trim_white_space(img_data,counters):
    for counter in counters:
        startY = 0
        endY = 30
        for y in range(counter[0][1],counter[1][1]):
            for x in range(counter[0][0],counter[1][0]):
                if img_data[y][x] == 0:
                    if not startY:
                        startY = y
                    endY = y

        counter[0][1] = max(startY-1,0)
        counter[1][1] = min(endY+1,30)

    return counters



def clear_one_pix(img_data):
    row_count = 0
    col_count = 0
    for row in img_data:
        max_connected_pix = 0
        col_count = 0
        for col in row:
            if col != 0:
                if max_connected_pix < 2:
                    for i in range(col_count - max_connected_pix,col_count):
                        img_data[row_count][i] = 255
                max_connected_pix = 0
            else :
                max_connected_pix += 1

            col_count += 1

        if max_connected_pix < 2:
            for i in range(col_count - max_connected_pix,col_count):
                img_data[row_count][i] = 255

        row_count += 1


def prep_img(img):
    # img as RGBA 30xFULL_IMG_Wx4
    img = color_quantization(img)

    pix_dict = {}

    for row in img:
        for col in row:
            color = get_color_str(col)

            if not pix_dict.get(color):
                pix_dict[color] = 0
            pix_dict[color] += 1

    color_arr = sorted(pix_dict.items(), key=lambda d: d[1], reverse=True)
    color_arr = color_arr[1:]

    
    color_dict = {}
    color_hitmap = {}

    for color in color_arr:
        if color[1] > 10:
            color_hitmap[color[0]] = True
            color_dict[color[0]] = np.zeros([30,FULL_IMG_W],dtype=np.uint8)
            color_dict[color[0]].fill(255)



    row_count = 0
    col_count = 0
    for row in img:
        col_count = 0
        for col in row:
            cell_color = get_color_str(col)
            if color_hitmap.get(cell_color):
                color_dict[cell_color][row_count][col_count] = 0
            col_count += 1
        row_count += 1

    for color_img in color_dict:
        img_data = color_dict[color_img]

        contours = find_contours(img_data, 0.5)

        contours = [[
            [int(floor(min(contour[:, 1]))), int(floor(min(contour[:, 0])))], # top-left point
            [int(ceil(max(contour[:, 1]))), int(ceil(max(contour[:, 0])))]  # down-right point
          ] for contour in contours]

        for contour in contours:
            w = contour[1][0] - contour[0][0] # width
            h = contour[1][1] - contour[0][1] # height
            area = w * h
            if area > 4:
                contour_filter(img_data,contour,area)


    result = np.zeros([30,FULL_IMG_W],dtype=np.uint8)
    result.fill(255)

    for color_name in color_dict:
        color_img = color_dict[color_name]
        row_count = 0
        col_count = 0
        for row in color_img:
            col_count = 0
            for col in row:
                if col == 0:
                    result[row_count][col_count] = 0
                col_count += 1
            row_count += 1

    clear_one_pix(result)

    result = result.T

    clear_one_pix(result)

    result = result.T

    return result



def split_letters(image, num_letters=4, debug=False):
    '''
    split full captcha image into `num_letters` lettersself.
    return list of letters binary image (0: white, 255: black)
    '''

    binary = prep_img(image)

    # find contours
    contours = find_contours(binary, 0.5)

    contours = [[
        [int(floor(min(contour[:, 1]))), 0], # top-left point
        [int(ceil(max(contour[:, 1]))), 30]  # down-right point
      ] for contour in contours]

    # keep letters order
    contours = sorted(contours, key=lambda contour: contour[0][0])

    # find letters box
    trimed_contours = []

    for contour in contours:
        if len(trimed_contours) > 0 and contour[0][0] < trimed_contours[-1][1][0] - 5:
            # skip inner contour
            continue
        trimed_contours.append(contour)

    trimed_contours = merge_counters(trimed_contours,num_letters)

    letter_boxs = []
    for contour in trimed_contours:
        # extract letter boxs by contour
        boxs = split_counter(binary, contour)
        for box in boxs:
            letter_boxs.append(box)

    letter_boxs = merge_counters(letter_boxs,num_letters)


    # check letter outer boxs number
    if len(letter_boxs) < num_letters:
        print('ERROR: number of letters is NOT valid', len(letter_boxs))
        # if debug:
        #     print(letter_boxs)
        #     plt.imshow(binary, interpolation='nearest', cmap=plt.cm.gray)
        #     for [x_min, y_min], [x_max, y_max] in letter_boxs:
        #         plt.plot(
        #             [x_min, x_max, x_max, x_min, x_min],
        #             [y_min, y_min, y_max, y_max, y_min],
        #             linewidth=2)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.show()
        return None
    elif len(letter_boxs) > num_letters:
        #print('ERROR: number of letters is NOT valid fixing', len(letter_boxs))
        
        letter_boxs = trim_counters(letter_boxs,num_letters)

        if len(letter_boxs) != num_letters:
            return None

    letter_boxs = trim_white_space(binary,letter_boxs)

    letters = []
    #plt.imshow(binary, interpolation='nearest', cmap=plt.cm.gray)
    for [x_min, y_min], [x_max, y_max] in letter_boxs:
        # plt.plot(
        #     [x_min, x_max, x_max, x_min, x_min],
        #     [y_min, y_min, y_max, y_max, y_min],
        #     linewidth=2)
        letter = resize(binary[y_min:y_max, x_min:x_max], LETTER_SIZE)
        letter = img_as_ubyte(letter < 0.6)
        letters.append(letter)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()


    #return None
    return letters

def merge_counters(contours,maxLength):
    if len(contours) <= maxLength :
        return contours

    new_contours = []
    index = 0
    while index < len(contours):
        letter_box = contours[index]
        w = letter_box[1][0] - letter_box[0][0]
        if w < 14 and index < (len(contours) -1):
            next_letter_box = contours[index + 1]
            next_w = next_letter_box[1][0] - next_letter_box[0][0]
            dist_w = next_letter_box[0][0] - letter_box[1][0]
            if next_w < 14 and dist_w < 2:
                #merge
                new_contours.append([[letter_box[0][0], letter_box[0][1]], [next_letter_box[1][0], next_letter_box[1][1]]])
                index += 2
            else :
                new_contours.append(letter_box)
                index += 1
        else :
            new_contours.append(letter_box)
            index += 1

    return new_contours


def trim_counters(letter_boxs,maxLength):
    if len(letter_boxs) <= maxLength :
        return letter_boxs

    tmp_letter_boxs = []
    tmp_letter_boxs2 = []
    new_letter_boxs = []
    index = 0

    for letter_box in letter_boxs:
        w = letter_box[1][0] - letter_box[0][0] # width
        h = letter_box[1][1] - letter_box[0][1] # height
        area = w*h
        tmp_letter_boxs.append([index,area])
        index += 1

    tmp_letter_boxs.sort(key=sortByArea,reverse=True)

    for num in range(0,maxLength):
        tmp_letter_boxs2.append(tmp_letter_boxs[num])

    tmp_letter_boxs2.sort(key=sortByIndex)

    for letter_box in tmp_letter_boxs2:
        new_letter_boxs.append(letter_boxs[letter_box[0]])

    return new_letter_boxs


def split_counter(binary, contour):
    boxs = []
    w = contour[1][0] - contour[0][0] # width
    h = contour[1][1] - contour[0][1] # height
    if w < 4 or h < 4 or w*h < 16:
        # skip too small contour (noise)
        return boxs
        

    if w < 24 :
        boxs.append(contour)
    else:
        # split 2 letters if w is large
        img_data = binary.T

        pix_count_Arr = []


        for i in range(contour[0][0],contour[1][0]):
            pix_count = 0
            for j in range(contour[0][1],contour[1][1]):
                if img_data[i][j] == 0:
                    pix_count += 1
            pix_count_Arr.append(pix_count)

        row_count = contour[0][0]
        last_split = contour[0][0]
        start_drop = False
        last_val = 0
        for val in pix_count_Arr:
            if val < last_val:
                start_drop = True
            else :
                if start_drop :
                    if row_count - last_split > 14:
                        boxs.append([[last_split,0],[row_count,30]])
                        last_split = row_count
                    start_drop = False
            last_val = val
            row_count += 1

        if last_split != row_count:
            boxs.append([[last_split,0],[row_count,30]])


    return boxs
