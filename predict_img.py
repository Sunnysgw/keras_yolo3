# coding: utf-8
import colorsys
import time
import numpy as np
import keras.backend as k
from keras.layers import Input
from model.utils import anchors, classes
from model.dark_net import yolo_body
from model.model import yolo_head


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    开始转换了
    :param feats:yolo模型输出的数据
    :param anchors:
    :param num_classes:
    :param input_shape:
    :param image_shape:
    :return:
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)



def yolo_evel(yolo_output, anchors, num_classes, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
    """
    对yolo网络的输出进行处理，转化为人能看懂的东西
    :param yolo_output:yolo网络的输出
    :param anchors:
    :param num_classes:
    :param image_shape:输入图像的shape
    :param max_boxes:一张图片最大的预测框数量
    :param score_threshold:评分阈值
    :param iou_threshold:iou的阈值
    :return:
    """
    # 获取yolo网络的输出层数
    num_layers = len(yolo_output)
    # 每个层对应的anchors的标签
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # 存放所有的框
    boxes = []
    # 存放所有的框对应的数值
    box_scores = []
    # 遍历每一个输出层
    for i in range(num_layers):
        # 这里就是转化的过程了



def generate_output_msg(model_path):
    """
    根据yolo网络的输出生成对图像的预测
    :param model_path:要加载的yolo模型
    :return:
    """
    num_anchors = len(anchors)
    num_classes = len(classes)
    # 构建yolo模型
    yolo_model = yolo_body(Input((None, None, 3)), num_anchors // 3, num_classes)
    # 不太清楚skip_mismatch这个选项要不要加
    yolo_model.load_weights(model_path)
    # 下面就是生成每种类别对应的颜色信息
    # 每个种类生成对应的颜色
    colors = [colorsys.hsv_to_rgb(x / num_classes, 1., 1.) for x in range(num_classes)]
    # 转化为数组，并将数值转化为8位
    colors = [np.array(x)*255. for x in colors]
    # 随机打乱颜色
    np.random.seed(int(time.time()))
    np.random.shuffle(colors)
    # 输入图片的shape
    input_image_shape = k.placeholder((2, ), name='input_image')
    boxes, scores, predict_classes = yolo_evel()


