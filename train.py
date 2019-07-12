# coding: utf-8
import time
import os
import platform
import shutil
import keras
import itchat
import numpy as np
import keras.backend as k
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model.model import yolo_loss
from model.dark_net import yolo_body
from model.utils import anchors, yolo_input_image_shape, windows_train_annotation_path, windows_val_annotation_path, \
    ubuntu_train_annotation_path, ubuntu_val_annotation_path, log_dir, classes, data_generator, batch_size, \
    get_train_data
import logging

logging.basicConfig(level=logging.INFO)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # itchat.login()
        itchat.auto_login()
        itchat.send_msg('train begin', 'filehelper')

    #        self.cap = cv2.VideoCapture(0)
    #        f, frame = self.cap.read()
    #        cv2.imwrite('demo.png', frame)
    #        itchat.send_image('demo.png', toUserName='filehelper')

    def on_epoch_end(self, epoch, logs=None):
        itchat.send_msg('epoch:%s-loss:%s-val_loss:%s' % (epoch, logs['loss'], logs['val_loss']), 'filehelper')


#        f, frame = self.cap.read()
#        cv2.imwrite('demo.png', frame)
#        itchat.send_image('demo.png', toUserName='filehelper')


def create_model(input_shape, anchors_, num_classes, load_pretrained=True, freeze_body=2, weight_path=None):
    """
    构建一个完整的用于训练的model，感觉这里还是有点奇葩的
    :param input_shape: 输入的宽和高
    :param anchors_: 构建model过程中用到的候选框
    :param num_classes: 训练数据集中物体类别总数
    :param load_pretrained: 是否加载预训练模型
    :param freeze_body: 选择暂时不训练的部分
    :param weight_path: 预训练模型路径
    :return:
    """
    # 构建模型的开端，清空前面的状态
    k.clear_session()
    # 设置显存动态增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    # 获取输入的shape
    h, w = input_shape
    # 定义输入,原文中没有明确定义输入的shape，只是定义输入的通道数是3
    img_input = Input((h, w, 3), dtype='float32')
    # 一共的anchor数目
    num_anchors = len(anchors_)
    # 定义标签输入，就是一个长度为3的list，长度为3是因为yolo网络预测的输出层是3个，这里num_anchors除以3是因为每一层用了
    # 3个anchor
    y_true = [Input(shape=(h // l, w // l, num_anchors // 3, num_classes + 5), dtype='float32') for l in [32, 16, 8]]
    # 构建用于预测的模型，输入是图片数据，其中的anchor数目和类别数目是用于决定输出层最后输出的filter数目
    model_body = yolo_body(img_input, num_anchors // 3, num_classes)
    # 这里我先把加载预训练模型这一块跳过去
    # 直接开始定义最后一层，loss，这也是训练这一块整个model的最后一层
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={
                            'anchors': anchors_,
                            'num_classes': num_classes,
                            'ignore_thresh': 0.5
                        })([*model_body.output, *y_true])
    # 至此，整个训练模型搭建完毕
    model = Model([img_input, *y_true], model_loss)
    return model


def train_yolo():
    """
    使用自己搭建的model进行训练
    :return:
    """
    # 获取检测数据集中的类别数目
    num_classes = len(classes)
    # 创建model
    model = create_model(yolo_input_image_shape, anchors, num_classes)
    # 日志文件路径
    # tar_log_dir = os.path.join(log_dir, time.strftime('%m_%d_%H_%M_%S'))
    tar_log_dir = os.path.join(log_dir, '000')
    # if os.path.exists(tar_log_dir):
    #     # 如果日志文件夹存在，删除之
    #     shutil.rmtree(tar_log_dir)
    # # 创建日志文件夹
    # os.mkdir(tar_log_dir)
    # 每轮末给微信发训练信息
    itchat_logging = LossHistory()
    logging_ = TensorBoard(log_dir=tar_log_dir)
    # 每三轮保存一次其中定义了epoch loss val loss的格式 03d表示是三位的整数
    checkpoint = ModelCheckpoint(os.path.join(tar_log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # 如果连续三轮，val loss都不降低，那么学习率变为原来的0.1
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # 如果两虚10轮val loss都不降低，那么停止学习
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    # 下面就是定义数据输入通道
    # 分别定义训练数据和测试数据
    # 根据操作系统选择要使用的数据
    if 'Linux' == platform.system():
        train_annotation_path = ubuntu_train_annotation_path
        val_annotation_path = ubuntu_val_annotation_path
    else:
        train_annotation_path = windows_train_annotation_path
        val_annotation_path = windows_val_annotation_path
    with open(train_annotation_path, 'r') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, 'r') as f:
        val_lines = f.readlines()
    all_lines = train_lines + val_lines
    # 打乱输入数据
    np.random.seed(int(time.time()))
    np.random.shuffle(all_lines)
    # 测试比例
    val_split = 0.05
    num_val = int(len(all_lines) * val_split)
    num_train = len(all_lines) - num_val
    # 这里先训练全部model，loss这里就是开玩笑的，因为model的最后一层就把loss的活都给干了
    model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    # 开始训练
    # model.load_weights(os.path.join(tar_log_dir, 'ep018-loss17.273-val_loss18.555.h5'))
    logging.info('Train on {} samples, val on {} samples, with batch size {}'.format(num_train, num_val, batch_size))
    # 一次性获取所有训练数据，这种方式占用内存空间过大
    # train_data, label_data = get_train_data(all_lines[:100], image_shape, anchors, num_classes)
    # 这里使用fit进行训练
    # model.fit(train_data, label_data, batch_size, epochs=50, callbacks=[logging_, checkpoint], validation_split=0.1)
    # 这里直接采用data generator生成数据流，一步一步生成数据，不占用过多内存
    model.fit_generator(data_generator(all_lines[:num_train], batch_size, yolo_input_image_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(all_lines[-num_val:], batch_size, yolo_input_image_shape, anchors,
                                                       num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[logging_, checkpoint])

    # 这里先训练全部model，loss这里就是开玩笑的，因为model的最后一层就把loss的活都给干了
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    # 开始训练
    logging.info('Train on {} samples, val on {} samples, with batch size {}'.format(num_train, num_val, batch_size))

    model.fit()
    model.fit_generator(data_generator(all_lines[:num_train], batch_size, yolo_input_image_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(all_lines[-num_val:], batch_size, yolo_input_image_shape, anchors,
                                                       num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[logging_, checkpoint, itchat_logging, reduce_lr, early_stopping])
    # 这里保存了两次，感觉第一次应该是可以的
    model.save(os.path.join(tar_log_dir, 'yolo_model.h5'))
    model.save_weights(os.path.join(tar_log_dir, 'yolo_weight.h5'))


if __name__ == '__main__':
    train_yolo()
