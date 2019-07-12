# coding: utf-8
import keras.backend as k
import tensorflow as tf
import numpy as np
import logging


logging.basicConfig(level=logging.DEBUG)


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    把yolo网络的输出转化为相应的预测
    :param feats:
    :param anchors:
    :param num_classes:
    :param input_shape:
    :param calc_loss:
    :return:
    """
    # 候选框数量
    num_anchors = len(anchors)
    # 把输入的候选框信息标准化
    anchors_tensor = k.reshape(k.constant(anchors), [1, 1, 1, num_anchors, 2])
    # 得到feature map 的高宽
    grid_shape = k.shape(feats)[1:3]
    # 构建一个完整的特征图的坐标系
    # 先整一个纵向的，然后横向扩充
    grid_y = k.tile(k.reshape(k.arange(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    # 之后是一个横向的，然后纵向扩充
    grid_x = k.tile(k.reshape(k.arange(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    # 把x和y合并起来，这里是先x后y，这样就是和矩阵的索引值刚好反过来，后面有进行缩放操作的时候，要把shape反过来
    grid = k.concatenate([grid_x, grid_y])
    # 转换数据类型
    grid = k.cast(grid, k.dtype(feats))
    # 把模型的shape重新规划一下
    feats = k.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, (5 + num_classes)])
    # 下面这一部分看起来就是莫名其妙的
    # 把x y坐标转化为占原始表格的比例，应该是为了减弱不同尺度目标对loss的影响
    # x y 直接是预测值加上原始的候选框的坐标
    box_xy = (k.sigmoid(feats[..., :2]) + grid) / k.cast(grid_shape[::-1], k.dtype(feats))
    # 宽长值就是原始值乘以预测值的指数，也是一个缩放的值，这里是宽长，所以要把原来的shape反过来
    box_wh = k.exp(feats[..., 2:4]) * anchors_tensor / k.cast(input_shape[::-1], k.dtype(feats))
    # 得到是否包含物体的概率
    box_confidence = k.sigmoid(feats[..., 4:5])
    # 得到对每个物体的预测概率
    box_class_probs = k.sigmoid(feats[..., 5:])
    # 如果是用于计算loss，返回的值就不太一样
    if calc_loss:
        # 返回特征图的坐标tensor，预测的值，预测的x y坐标，预测的宽高
        return grid, feats, box_xy, box_wh
    # 不计算loss，直接返回的就是预测的x y 坐标，预测的宽高，是否包含物体的概率，预测的类别概率
    return box_xy, box_wh, box_confidence, box_class_probs


def box_iou(pred_info, true_box):
    """
    计算预测信息和真实信息之间的偏差，用iou表示，相当于计算每个预测框和所有有效的标签框之间的iou的值，返回的shape如下
    :param pred_info:shape（13，13，3，4）
    :param true_box:shape（j，4）
    :return:shape（13, 13, 3, j）
    """
    # 第一步，处理预测框，得到左上角和右下角两个点的坐标
    # 这里扩充维度是为了方便和下面预测的值作比较，意思就是这里(1, 2)的维度和标签的(j, 2)比较
    pred_info = k.expand_dims(pred_info, -2)
    pred_xy = pred_info[..., :2]
    pre_wh = pred_info[..., 2:4]
    pre_wh_half = pre_wh / 2.
    pre_min = pred_xy - pre_wh_half
    pre_max = pred_xy + pre_wh_half
    # 第二步，同样的手段，得到真值左上角和右下角两点坐标
    true_box = k.expand_dims(true_box, 0)
    true_xy = true_box[..., :2]
    true_wh = true_box[..., 2:4]
    true_wh_half = true_wh / 2.
    true_min = true_xy - true_wh_half
    true_max = true_xy + true_wh_half
    # 下面就是计算相应的iou
    intersect_min = k.maximum(pre_min, true_min)
    intersect_max = k.minimum(pre_max, true_max)
    intersect_wh = k.maximum(intersect_max - intersect_min, 0.)
    # 计算得到相交部分的面积
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # 分别计算得到预测框和标签框的面积
    pre_area = pre_wh[..., 0] * pre_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    # 计算得到iou
    iou = intersect_area / (pre_area + true_area - intersect_area)
    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """
    定义yolo网络的loss
    :param args: 配合keras中的自定义层，接受传递进来的tensor
    :param anchors: 用到的anchor
    :param num_classes: 预测的类别数目
    :param ignore_thresh: 定义的是否抛弃anchor框的阈值
    :param print_loss: 是否打印loss
    :return:
    """
    # 定义输出的层的数目，y1、y2、y3每个都会输出num_layers个层，每层分别针对对应的anchor预测
    num_layers = len(anchors) // 3
    # 得到标签数据，感觉原版的不太好，就按照我的来了
    y_true = args[3:]
    logging.debug(y_true[0].shape)
    # 得到yolo最后的输出层
    yolo_output = args[:3]
    logging.debug(yolo_output[0].shape)
    # 这里确定每个yolo的输出层对应的anchor的id
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    # 得到输入图片的维度，yolo第一层的输出的宽高乘以32，得到输入图像的宽高
    input_shape = k.cast(k.shape(yolo_output[0])[1:3] * 32, k.dtype(y_true[0]))
    # 得到三个feature map的shape
    grid_shape = [k.cast(k.shape(yolo_output[i])[1:3], k.dtype(y_true[0])) for i in range(num_layers)]
    # 初始化loss
    loss = 0
    # 得到batch的size
    m = k.shape(yolo_output[0])[0]
    # 类型转化，转化为浮点类型
    mf = k.cast(m, k.dtype(yolo_output[0]))
    # 逐层计算其中的loss
    for layer_id in range(num_layers):
        # 从y_true中得到定义的标签
        object_mask = y_true[layer_id][..., 4:5]
        true_class_probs = y_true[layer_id][..., 5:]
        # 把yolo网络的输出进行包装，得到相应的预测值，计算loss的标志位置为True
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_output[layer_id],
                                                     anchors[anchor_mask[layer_id]],
                                                     num_classes,
                                                     input_shape,
                                                     calc_loss=True)
        # 把x y w h 连接起来
        pred_box = k.concatenate([pred_xy, pred_wh])
        # 下面对输入的标签进行相应的处理，这里还是不太清楚，要把数据输入那一部分写完之后才可以
        # 因为原来的标签是一个比例，所以要乘以shape
        # TODO 这个减去grid是几个意思呀
        raw_true_xy = y_true[layer_id][..., :2] * grid_shape[layer_id][::-1] - grid
        # TODO 怎么还除以anchor的大小，乘以特征图的大小
        raw_true_wh = k.log(y_true[layer_id][..., 2:4] / anchors[anchor_mask[layer_id]] * input_shape[::-1])
        # TODO 就很奇怪，这个意思是object_mask中但凡是预测值为0的，wh值也变为0
        raw_true_wh = k.switch(object_mask, raw_true_wh, k.zeros_like(raw_true_wh))
        # TODO 这个也是看过数据生成那一部分才会明白
        box_loss_scale = 2 - y_true[layer_id][..., 2:3] * y_true[layer_id][..., 3:4]
        # 标志是否忽略之，动态增长的一个数组，这里是特征图每个预测位置的每个对应的anchor有一个标签
        ignore_mask = tf.TensorArray(k.dtype(y_true[0]), size=1, dynamic_size=True)
        # 把预测标签转化为布尔值
        object_mask_bool = k.cast(object_mask, 'bool')

        # 下面就是循环更新ignore_mask

        def loop_body(b, ignore_mask_):
            """
            定义循环体，循环更新ignore_mask，这个是一个循环处理一张图片在该特征图中的所有标注框
            :param b:处理的图片在一个batch中的id
            :param ignore_mask_:
            :return:
            """
            # 只选择有物体的标签框
            true_box = tf.boolean_mask(y_true[layer_id][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # 计算一个图片中所有物体和候选框之间的iou
            iou = box_iou(pred_box[b], true_box)
            # 得到最佳的iou
            best_iou = k.max(iou, -1)
            # 更新ignore_mask，更新对于每一个锚点的评价
            ignore_mask_ = ignore_mask_.write(b, k.cast(best_iou < ignore_thresh, k.dtype(true_box)))
            return b + 1, ignore_mask_

        # 定义循环更新ignore_mask，其中第一个参数是条件函数，第二个参数是循环函数，两个函数使用相同的输入参数，怎么都
        # 感觉和一个for循环一个效果
        # TODO 查了资料中间说，计算图中不能实现for循环，所以需要这个while_loop，就很让人困惑
        _, ignore_mask = k.control_flow_ops.while_loop(lambda b, *a: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # 扩充维度，前面前面的max操作会降维
        ignore_mask = k.expand_dims(ignore_mask, -1)
        # 下面就是计算loss
        # 计算坐标loss，只计算其中有物体的那些坐标的loss
        xy_loss = object_mask * box_loss_scale * k.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        # 计算宽高loss
        wh_loss = object_mask * box_loss_scale * 0.5 * k.square(raw_true_wh - raw_pred[..., 2:4])
        # 计算搜索框loss
        confidence_loss = object_mask * k.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (
                    1 - object_mask) * k.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                             from_logits=True) * ignore_mask
        # 计算类别loss
        class_loss = object_mask * k.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)
        xy_loss = k.sum(xy_loss) / mf
        wh_loss = k.sum(wh_loss) / mf
        confidence_loss = k.sum(confidence_loss) / mf
        class_loss = k.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
    return loss
