# 验证码类型2

[Download](https://github.com/scientihark/tensorCaptcha/releases/tag/type2.v1)

## 说明

图例  ![img](example.png)

字符包括 0-9 A-H J-K M-Z

字数 4

包含线条混淆

### 依赖

node 依赖

```

npm i python-shell 

```

python 依赖

```

numpy

scikit-image

tensorflow

matplotlib

sklearn

```

### A.直接用训练结果

可以同样处理部分数字验证码

下载训练结果

[Download](https://github.com/scientihark/tensorCaptcha/releases/tag/type2.v1)


命令行调用 

```

python predict.py --fname [filePath]

```

nodejs 

```

const OCR = require('./captchaOCR.js');

OCR(`${filePath}`,(code)=>{
	// success callback
},(error)=>{
	// fail callback
})

```

### B.从头开始构建并训练

下载原始数据

[Download](https://github.com/scientihark/tensorCaptcha/releases/tag/type2.v1)


```

node makeTrainData.js

python createTrainData.py

python train.py

```

训练完成后会打印成功率

我们还提供了简单的验证程序对原始图片进行识别准确率验证


```

node checkTrainResult.js

```


## 解法思路

### 获得样例图片

由于初期并没有足够的样例图片，我们有2种思路来得到样例

1.人工/打码平台 打码

2.生成相似验证码训练后尝试识别，以高错误率为代价获取正确识别结果为样例

这里我们2者都采用，得到 1.4W 的样例图片

整个过程就是人工打码100个+生成图片1W -> 训练 -> 测试并保存正确结果 -> 人工对部分错误结果打码 -> 训练 ...

下面附上生成相似验证码图片的代码 （字体请自行寻找）

```

const { createCanvas,createImageData,registerFont,loadImage } = require('canvas')

registerFont('fonts/XXXXX.ttf', { family: 'XXXXX' })

function rand(l){
	return Math.floor(Math.random() * l)
}

function randCaptchaStr(pattern,length){
	let out = "",i;
    for( i=0; i<length; i++){
        out+= pattern[rand(pattern.length)]
    }
  	return out;
}

function drawLine(ctx){
	ctx.beginPath();
	ctx.fillStyle = COLORS[rand(COLORS.length)];
	
	ctx.moveTo(200 - rand(200), 40 - rand(40));
	ctx.bezierCurveTo(200 - rand(200), 40 - rand(40), 200 - rand(200), 40 - rand(40), 200 - rand(200), 40 - rand(40));
	ctx.stroke();
	ctx.closePath();
}

const FONTS = [
	'22px "XXXXX" bold 400',
]

const COLORS = [
	"#576c49",
	"#4b3e4c",
	"#424359",
	"#6f7f90",
	"#008100",
	"#0000ff"
]

let captchaStr = randCaptchaStr("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",4)

let canvas = createCanvas(85, 30),
	ctx = canvas.getContext('2d'),
	letters = captchaStr.split(''),
	i,
	x = 5 + rand(5);

// white bg
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, 85, 30);

ctx.fillStyle = 'black';
ctx.lineWidth = 0.2

let lineCount = 3 //(10+rand(10))

for(i=0;i<lineCount;i++){
	drawLine(ctx)
}

for(i=0;i<letters.length;i++){
	let letter = letters[i],
		font = FONTS[rand(FONTS.length)],
		y = 20 + rand(8);
	
	ctx.font = font;
	ctx.fillStyle = COLORS[rand(COLORS.length)];

	ctx.fillText(letter, x, y);

	x += (18 + rand(5))
}

let result = canvas.toBuffer();
fs.writeFileSync(`captcha.png`,result);


```

然后就可以生成训练数据了

### 1.颜色聚合

原始图片文字边缘有很多像素噪点，通过合并颜色来简单过滤一些

这里，我们使用 KMeans 

```

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

```

### 2.生成分色彩的矩阵

将颜色按像素量排序，然后去掉最多的白色

```

pix_dict = {}

for row in img:
    for col in row:
        color = get_color_str(col)

        if not pix_dict.get(color):
            pix_dict[color] = 0
        pix_dict[color] += 1

color_arr = sorted(pix_dict.items(), key=lambda d: d[1], reverse=True)
color_arr = color_arr[1:]

```

然后每个像素总量大于10的颜色生成一个用255填充的矩阵

```

color_dict = {}
color_hitmap = {}

for color in color_arr:
    if color[1] > 10:
        color_hitmap[color[0]] = True
        color_dict[color[0]] = np.zeros([30,FULL_IMG_W],dtype=np.uint8)
        color_dict[color[0]].fill(255)

```

然后遍历原图，分离颜色

```

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

```

### 2.对色彩矩阵去除无用色块

对每个颜色取 contours

```

img_data = color_dict[color_img]

contours = find_contours(img_data, 0.5)

contours = [[
            [int(floor(min(contour[:, 1]))), int(floor(min(contour[:, 0])))], # top-left point
            [int(ceil(max(contour[:, 1]))), int(ceil(max(contour[:, 0])))]  # down-right point
          ] for contour in contours]

```

面积大于4个像素的contour进行检查

```

for contour in contours:
    w = contour[1][0] - contour[0][0] # width
    h = contour[1][1] - contour[0][1] # height
    area = w * h
    if area > 4:
        contour_filter(img_data,contour,area)

```

这里用单位面积的像素值来进行过滤

统计了一些图片的数据，对于包含文字的contour 这个值一定大于0.15

同时 宽高比超过2的横向图这里也进行简单的过滤

```

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

```

### 3.拼合颜色矩阵

把不同颜色的矩阵简单拼在一起

```

result = np.zeros([30,85],dtype=np.uint8)
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

```

### 4.去除细线

拼合后还可能有仅1像素的细线

这里进行去除

```

clear_one_pix(result)

result = result.T

clear_one_pix(result)

result = result.T

```

```

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

```

### 5.对拼合后图片取contours

注意，由于文字不存在上下的位置关系，这里直接将Y的范围扩大到整个高度，来简单处理contour只取到文本上下某部分的问题

```

contours = find_contours(binary, 0.5)

contours = [[
        [int(floor(min(contour[:, 1]))), 0], # top-left point
        [int(ceil(max(contour[:, 1]))), 30]  # down-right point
      ] for contour in contours]

contours = sorted(contours, key=lambda contour: contour[0][0])

trimed_contours = []

for contour in contours:
    if len(trimed_contours) > 0 and contour[0][0] < trimed_contours[-1][1][0] - 5:
        # skip inner contour
        continue
    trimed_contours.append(contour)

```

### 6.分割/合并部分contours

```

trimed_contours = merge_counters(trimed_contours,num_letters)

```

把2个宽度小于14且相距小于2像素的contour合并

```

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


```

然后做分割

```

letter_boxs = []
for contour in trimed_contours:
    # extract letter boxs by contour
    boxs = split_counter(binary, contour)
    for box in boxs:
        letter_boxs.append(box)

```

对宽度大于24的counter进行分割

这里用的是纵向像素值进行分割

先计算每一列黑色像素的数量

然后找出像素数量下降的位置

用这个位置和距离上一个切割位置的距离 > 14 来 进行切割

```

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

```

然后我们再合并一次

防止出现过度切割

```

letter_boxs = merge_counters(letter_boxs,num_letters)

```

### 7.判断counters数量，对过多counters进行trim

```

if len(letter_boxs) < num_letters:
    print('ERROR: number of letters is NOT valid', len(letter_boxs))
    return None
elif len(letter_boxs) > num_letters:
    letter_boxs = trim_counters(letter_boxs,num_letters)

    if len(letter_boxs) != num_letters:
        return None

```

对counters按面积排序

只保留指定个数的counters

```

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

```

### 8.对counters上下白边进行trim

```

letter_boxs = trim_white_space(binary,letter_boxs)

```

```

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

```

### 9.图片按最终counters切割

```

letters = []
for [x_min, y_min], [x_max, y_max] in letter_boxs:
    letter = resize(binary[y_min:y_max, x_min:x_max], LETTER_SIZE)
    letter = img_as_ubyte(letter < 0.6)
    letters.append(letter)

```

整体实现可以参见代码

切片完毕后 就可以开始使用tensorflow训练啦

### tensorflow模型

一个多层的CNN

```

Input Layer （16, 14）

Convolutional Layer #1 	filters=32
      					kernel_size=[3, 3]
      					padding='same'
      					activation=tf.nn.relu

Pooling Layer #1		pool_size=[2, 2]
						strides=2

Convolutional Layer #2 	filters=64
      					kernel_size=[3, 3]
      					padding='same'
      					activation=tf.nn.relu

Pooling Layer #2		pool_size=[2, 2]
						strides=2

Convolutional Layer #3 	filters=128
      					kernel_size=[3, 3]
      					padding='same'
      					activation=tf.nn.relu

Dense Layer #1			units=512
						activation=tf.nn.relu

Dropout #1				rate=0.4

Dense Layer #2			units=512
						activation=tf.nn.relu

Dropout #2				rate=0.4

Logits Layer			units=75   			# 0-9A-Za-z  48 = '0' 122 = 'z' 


```

### 训练结果

batchSize=500 训练 2W个step 后

tensorflow 步骤的识别准确率为97.1%

由于存在切图失败的情况

切图步骤的成功率为99%

### 优化

切图步骤复杂度还比较高

某些步骤可能可以省略或者优化



