# 验证码类型1

## 说明

图例见example.gif

字符包括 0-9

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

```

### A.直接用训练结果

可以同样处理部分数字验证码

下载训练结果

见release


命令行调用 

```

magick convert [filePath] -density 300 -units PixelsPerInch -type Grayscale -negate -scale 200% -morphology Erode Octagon:1 -negate -morphology Erode disk:1 -scale 50% -threshold 99% -colors 2 -colorspace gray -background white -flatten [outputPath]

python predict.py --fname [outputPath]

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

见release


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

### 先对输入图片预处理

尽量去除线条并将图片二值化

这里偷懒没有用CV写，测试方便用了image Magic


```

magick convert [filePath] -density 300 -units PixelsPerInch -type Grayscale -negate -scale 200% -morphology Erode Octagon:1 -negate -morphology Erode disk:1 -scale 50% -threshold 99% -colors 2 -colorspace gray -background white -flatten [outputPath]

```

预处理思路

先灰度化 -> 2倍放大图像 -> 取负向 + 使用 morphology Erode Octagon:1 （去除细线） -> 再次取负向 + 使用 -morphology Erode disk:1 （尝试恢复部分原有较细的字符连接处） 

-> 缩放回原有大小 -> threshold 99% colors 2 colorspace gray background white flatten （黑白二值化）

### 生成训练数据

此验证码来源于生产环境的爬虫使用其他验证码识别方案反回的正确存档 （60W+）

我们对原图像进行上述预处理后就可以开始创建训练集了

首先是将原图切片

我们这里用 skimage 的 find_contours 方法

```

contours = find_contours(binary, 0.5)

```

需要拆分宽度超过2个字符的图块

并且 合并过小且相互邻接的图块

具体实现可以参见代码

切片完毕后 就可以开始使用tensorflow训练啦

### tensorflow模型

这里用的是标准的 CNN 就不赘述了

### 训练结果

训练了60W个样本

tensorflow 步骤的识别准确率为98.9%

由于存在切图失败的情况

包括切图步骤的识别准确率为97%

### 生产实战

该验证码识别用于生产环境爬虫后 爬虫统计的验证码识别准确率为 95% 

我们将失败的验证码存档后

选取100条人工设置了答案

并使用这些错误集重新进行了训练

训练 100 个step 后，重新对全部失败验证码进行识别

准确率提升到了75%

### 后记

由于目标网站在验证码识别上线后3天即发觉了验证码不再起到防护作用，修改了验证码（增加了混淆并变成了8位数字字母）

继续训练样本不足

此次试验中止


