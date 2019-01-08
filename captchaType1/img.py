from __future__ import print_function, absolute_import, division
from math import floor, ceil
from skimage import img_as_ubyte
from skimage.measure import find_contours
from skimage.util import crop
from skimage.transform import resize

import matplotlib.pyplot as plt

BINARY_THRESH = 30 # image binary thresh
LETTER_SIZE = (20, 15) # letter heigth,width

def sortByArea(elem):
    return elem[1]

def sortByIndex(elem):
    return elem[0]

def split_letters(image, num_letters=4, debug=False):
    '''
    split full captcha image into `num_letters` lettersself.
    return list of letters binary image (0: white, 255: black)
    '''

    # binarization
    binary = image > BINARY_THRESH

    # find contours
    contours = find_contours(binary, 0.5)

    contours = [[
        [int(floor(min(contour[:, 1]))), int(floor(min(contour[:, 0])))], # top-left point
        [int(ceil(max(contour[:, 1]))), int(ceil(max(contour[:, 0])))]  # down-right point
      ] for contour in contours]

    # keep letters order
    contours = sorted(contours, key=lambda contour: contour[0][0])

    # find letters box
    letter_boxs = []
    for contour in contours:
        if len(letter_boxs) > 0 and contour[0][0] < letter_boxs[-1][1][0] - 5:
            # skip inner contour
            continue
        # extract letter boxs by contour
        boxs = split_counter(binary, contour)
        for box in boxs:
            letter_boxs.append(box)

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
        
        letter_boxs = merge_counters(letter_boxs,num_letters)

        letter_boxs = trim_counters(letter_boxs,num_letters)

        if len(letter_boxs) != num_letters:
            return None

    letters = []
    #plt.imshow(binary, interpolation='nearest', cmap=plt.cm.gray)
    for [x_min, y_min], [x_max, y_max] in letter_boxs:
        # plt.plot(
        #     [x_min, x_max, x_max, x_min, x_min],
        #     [y_min, y_min, y_max, y_max, y_min],
        #     linewidth=2)
        letter = resize(image[y_min:y_max, x_min:x_max], LETTER_SIZE)
        # plt.imshow(letter, interpolation='nearest', cmap=plt.cm.gray)
        # plt.show()
        letter = img_as_ubyte(letter < 0.6)
        letters.append(letter)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    return letters

def merge_counters(contours,maxLength):
    if len(contours) <= maxLength :
        return contours

    new_contours = []
    index = 0
    while index < len(contours):
        letter_box = contours[index]
        w = letter_box[1][0] - letter_box[0][0]
        if w < 10 and index < (len(contours) -1):
            next_letter_box = contours[index + 1]
            next_w = next_letter_box[1][0] - next_letter_box[0][0]
            dist_w = next_letter_box[0][0] - letter_box[1][0]
            if next_w < 10 and dist_w < 1:
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
    if w < 5 or h < 5 or w*h < 50:
        # skip too small contour (noise)
        return boxs
        

    if w < 25 :
        boxs.append(contour)
    else:
        # split 2 letters if w is large
        x_mean = contour[0][0] + int(round(w / 2))
        sub_contours = [
            [contour[0], [x_mean, contour[1][1]]],
            [[x_mean, contour[0][1]], contour[1]]
        ]
        for [x_min, y_min], [x_max, y_max] in sub_contours:
            # fit y_min, y_max
            y_min_val = min(binary[y_min + 1, x_min:x_max])
            y_max_val = min(binary[y_max - 1, x_min:x_max])
            while y_min_val or y_max_val:
                if y_min_val:
                    y_min += 1
                    y_min_val = min(binary[y_min + 1, x_min:x_max])
                if y_max_val:
                    y_max -= 1
                    y_max_val = min(binary[y_max - 1, x_min:x_max])

            boxs.append([[x_min, y_min], [x_max, y_max]])

    return boxs
