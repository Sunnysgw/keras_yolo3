# coding: utf-8
import math
import pickle
import numpy as np
# from sklearn.cluster import KMeans
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# 训练批次
batch_size = 8
# 一共是9个anchor尺寸，这个是在coco那里通过聚类的方式得到的
anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
# yolo网络输入图片的尺寸
yolo_input_image_shape = (416, 416)
# 训练数据路径，这里是pascal_voc 2012数据集中的数据
windows_train_annotation_path = 'train_data/2012_train.txt'
ubuntu_train_annotation_path = 'train_data/2012_train_ubuntu.txt'
# 测试数据路径
windows_val_annotation_path = 'train_data/2012_val.txt'
ubuntu_val_annotation_path = 'train_data/2012_val_ubuntu.txt'
# 日志路径
log_dir = 'logs/'
# pascal_voc 2012 类别，一共是20类
classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


def rand(a=0., b=1.):
    """
    随机生成a到b之间的数据
    :param a:
    :param b:
    :return:
    """
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5,
                    val=1.5, proc_img=True):
    """
    对传进来的数据进行随机增强，这里的增强指的是对数据进行随机的缩放
    :param annotation_line:
    :param random:
    :param max_boxes:
    :param jitter:
    :param hue:
    :param sat:
    :param val:
    :param proc_img:
    :return:
    """
    line_data = annotation_line.split()
    # 得到图像数据
    image = Image.open(line_data[0])
    # 得到图像的宽和高
    iw, ih = image.size
    # 预计的宽和高
    h, w = yolo_input_image_shape
    # 提取出来box的数据，使用numpy的数组填充
    box = np.array([list(map(int, i.split(','))) for i in line_data[1:]])
    if not random:
        # 不使用数据增强，直接简单粗暴地缩放
        # 取图片期望宽高与实际宽高的比例的最小值，这样就是保证缩放之后的图片比期望宽高要小
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        # 得到需要平移的距离
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        # 处理图像，对图像进行缩放
        if proc_img:
            # 处理图像
            image = image.resize((nw, nh), Image.BICUBIC)
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            # 把缩放之后的图片粘贴到宽高为 w h 的背景板上
            new_img.paste(image, (dx, dy))
            # 把像素值归一化到0-1
            image = np.array(new_img) / 255.
        # 根据前面的缩放结果调整定位框
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            # 打乱输入数据
            np.random.shuffle(box)
            if len(box) > max_boxes:
                # 只取最大max_boxes个标签框去训练
                box = box[:max_boxes]
                box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
                box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
                box_data[:len(box)] = box
        # 就此结束，返回图片数据和标签数据
        return image, box_data
    # 这里就是使用随机的方式进行数据增强
    # 随机生成宽高比
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    # 随机生成缩放比例
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    # 根据新生成的宽高resize图片
    image = image.resize((nw, nh), Image.BICUBIC)
    # 这里放置新图片的位置也是随机生成的
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    # 新生成全是灰色的幕布
    new_img = Image.new('RGB', (w, h), (128, 128, 128))
    # 把缩放之后的图片放到幕布上去
    new_img.paste(image, (dx, dy))
    image = new_img
    # 决定是否反转图片，这个是左右反转
    flip = rand() < .5
    if flip:
        # 一半的概率左右翻转
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # 下面就是像素振动
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    # 把图片调整到HSV空间，h：色相 s：纯度 v：饱和度
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    # 把调整好的图像转化回来
    image = hsv_to_rgb(x)
    # 根据前面的缩放调整相应的标注框
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dx
        if flip:
            box[:, [0, 2]] = w - box[:, [0, 2]]
        box[:, 0: 2][box[:, 0: 2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        # 因为这里是随机缩放平移，肯定是会有一些东西被平移出整个屏幕的，平移出屏幕的特征就是 w h 变成0了
        # 这个很容易想清楚
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        if len(box) > max_boxes:
            box = box[: max_boxes]
        box_data[: len(box)] = box
    return image, box_data


def preprocess_true_boxes(true_box, input_shape, anchors_, num_classes):
    """
    把原始输入的数据转化为标签需要的格式[n, f_w, f_h, num_anchors, 5+num_classes]
    :param true_box:
    :param input_shape:
    :param anchors_:
    :param num_classes:
    :return:
    """
    assert (true_box[..., 4] < num_classes).all(), '类别标签必须是小于类别总数'
    # 预测的出口数量
    num_layers = len(anchors_) // 3
    # 每一层使用的anchor标志位
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    true_box = np.array(true_box, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # 得到标签框的中心坐标
    true_box_xy = (true_box[..., 0:2] + true_box[..., 2:4]) // 2
    # 得到标签框的宽高数据
    true_box_wh = true_box[..., 2: 4] - true_box[..., 0: 2]
    # 把标签框的数据转化为 x y w h 的形式
    true_box[..., 0: 2] = true_box_xy / input_shape[::-1]
    true_box[..., 2: 4] = true_box_wh / input_shape[::-1]
    # 得到batch_size
    m = true_box.shape[0]
    # 得到三个特征图的shape
    grid_shape = [input_shape // i for i in [32, 16, 8]]
    # 生成标准的占位符，这个就是正式的标签数据，下面的工作是填充这个东西
    y_true = [np.zeros((m, grid_shape[i][0], grid_shape[i][1], len(anchor_mask[i]), 5 + num_classes), dtype='float32')
              for i in range(num_layers)]
    # 扩充维度，用于计算下面的iou
    anchors_ = np.expand_dims(anchors_, 0)
    anchors_maxes = anchors_ / 2.
    anchors_mins = -anchors_maxes
    # 找到其中有效的标签数目，所谓有效，是因为对每个图片都是使用一个(20, 2)的全零矩阵初始化的
    valid_mask = true_box_wh[..., 0] > 0
    for b in range(m):
        # 把有效的对应出来
        wh = true_box_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # 扩充维度，后面好进行运算
        wh = np.expand_dims(wh, -2)
        # 这里就是运算照片中的每个物体和哪一个候选框最契合
        box_maxes = wh / 2.
        box_mins = -box_maxes
        intersect_mins = np.maximum(box_mins, anchors_mins)
        intersect_maxes = np.minimum(box_maxes, anchors_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchors_area = anchors_[..., 0] * anchors_[..., 1]
        # 计算iou，即（联合面积）/（物体面积+候选框面积-联合面积）
        iou = intersect_area / (box_area + anchors_area - intersect_area)
        # 找到契合度最高的候选框
        best_anchor = np.argmax(iou, axis=-1)
        # 下面就是找到该图片中和每个物体契合度最高的候选框的具体位置，并填充到y_true中去
        for t, n in enumerate(best_anchor):
            # 依次遍历每个物体，t：物体编号 n：候选框的编号
            for l in range(num_layers):
                # 依次遍历每个层
                if n in anchor_mask[l]:
                    # 如果候选框的编号在该层中
                    # 计算出是属于哪个框
                    i = np.floor(true_box[b, t, 0] * grid_shape[l][1]).astype('int32')
                    j = np.floor(true_box[b, t, 1] * grid_shape[l][0]).astype('int32')
                    # 找到具体属于哪个维度
                    k = anchor_mask[l].index(n)
                    c = true_box[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_box[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
    return y_true


def data_generator(annotation_lines, batch_size, input_shape, anchors_, num_classes):
    """
    定义数据生成器，这个地方有必要提升一下速度
    :param annotation_lines:要训练的数据
    :param batch_size:
    :param input_shape:
    :param anchors_:
    :param num_classes:
    :return:
    """
    # 数据总量
    n = len(annotation_lines)
    i = 0
    while True:
        # 存放图片数据
        img_data = []
        # 存放标签数据
        box_data = []
        for b in range(batch_size):
            if i == 0:
                # 如果从头开始，打乱数据集
                np.random.shuffle(annotation_lines)
            # 随机增强数据
            image, box = get_random_data(annotation_lines[i])
            img_data.append(image)
            box_data.append(box)
            # 保证i依次增长的同时不会越界
            i = (i + 1) % n
            # print('处理到第%s个数据' % b)
        img_data = np.array(img_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors_, num_classes)
        # 奇葩，把图片数据和标签同时作为输入，然后每次训练只需要看第一个就好了，后面那个0数组只是随意定义的
        # TODO 是不是有一种比这个更好一些的实现方式，这样就是浪费了一个位置，这样acc这个数值对于这个版本的yolo没有意义
        yield [img_data, *y_true], np.zeros(batch_size)


def get_train_data(annotation_lines, input_shape, anchors_, num_classes):
    """
    这里使用随机的数据生成的方式一次性生成所有的数据
    :param annotation_lines:要训练的数据
    :param input_shape:
    :param anchors_:
    :param num_classes:
    :return:
    """
    # 首先遍历annotation中的所有的行，生成对应数据
    # 打乱数据
    np.random.shuffle(annotation_lines)
    # 从标注数据中获取图片和标注框数据
    # img_box_data = [[get_random_data(line_data)] for line_data in annotation_lines]
    img_box_data = []
    for index, line_data in enumerate(annotation_lines):
        img_box_data.append(get_random_data(line_data))
        print('%s of %s' % (index, len(annotation_lines)))
    img_data, box_data = [i[0] for i in img_box_data], [j[1] for j in img_box_data]
    img_data = np.array(img_data)
    box_data = np.array(box_data)
    # 把标注数据进行进一步的处理
    y_true = preprocess_true_boxes(box_data, input_shape, anchors_, num_classes)
    # 构成训练数据，这里包括图片数据和每张图片对应的三个输出标签
    train_data = [img_data, *y_true]
    with open('cache/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    # 一共有的batch数目
    # batch_num = math.ceil(len(annotation_lines) / batch_size)
    # 把所有的train数据划分成能够训练的格式
    # train_data = [[train_data[0][i * batch_size: (i + 1) * batch_size],
    #                train_data[1][i * batch_size: (i + 1) * batch_size],
    #                train_data[2][i * batch_size: (i + 1) * batch_size],
    #                train_data[3][i * batch_size: (i + 1) * batch_size]] for i in range(batch_num)]
    # 定义label数据
    label_data = np.zeros((len(annotation_lines)))
    return train_data, label_data


# def k_means():
#     """
#     使用k_means对训练数据的宽高进行聚类，这里默认是聚9类
#     :return:
#     """
#     with open(ubuntu_train_annotation_path, 'r') as f:
#         train_lines = f.readlines()
#     with open(ubuntu_val_annotation_path, 'r') as f:
#         val_lines = f.readlines()
#     all_lines = train_lines + val_lines


if __name__ == '__main__':
    with open('../train_data/2012_train_ubuntu.txt', 'r') as f:
        lines = f.readlines()
    data, _ = data_generator(lines, 128, yolo_input_image_shape, anchors, len(classes))
