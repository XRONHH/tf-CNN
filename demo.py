#coding=utf-8

import tensorflow as tf
from input_data import read_data
# 数据文件夹
data_dir = "G:/org_img/change_img"

# 训练还是测试
train = True
#train = None
# 模型文件路径
model_path = "G:/model/child_image_model"

#将文件夹中数据读取并处理
fpaths, datas, labels = read_data(data_dir)

# 计算有多少类图片，作为输出的大小（维度）
num_classes = len(set(labels))

# 定义Placeholder，存放输入和标签
#设置训练图片为32*32
input_datas = tf.placeholder(tf.float32, [None, 128, 128, 3])
input_labels = tf.placeholder(tf.int32, [None])
# 存放DropOut参数的容器，训练时为0.25，测试时为0
# 使用DropOut参数，控制整个cnn网络的冗杂度，防止过拟合
dropout_placeholdr = tf.placeholder(tf.float32)

def create_conv(input_placeholder):
    
    # 定义卷积层0, 20个卷积核, 卷积核大小为5，使用激活函数Relu激活
    conv0 = tf.layers.conv2d(input_placeholder, 20, 5, activation=tf.nn.relu)
    # 定义池化层max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])
    # 定义卷积层1, 40个卷积核, 卷积核大小为4，使用激活函数Relu激活
    conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
    # 定义池化层max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
    return(pool0,pool1)
pool0,pool1 = create_conv(input_datas)

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool1)
# 全连接层，转换为长度为100的特征向量
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
# 加上DropOut，防止过拟合
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.arg_max(logits, 1)


# 利用交叉熵定义损失
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(input_labels, num_classes),
    logits=logits
)
# 平均损失
mean_loss = tf.reduce_mean(losses)

# 定义优化器，指定要优化的损失函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)


# 用于保存和载入模型
saver = tf.train.Saver()

with tf.Session() as sess:

    if train:
        print("----------训练模式-----------")
        # 如果是训练，初始化参数
        sess.run(tf.global_variables_initializer())
        # 定义输入和Label以填充容器，训练时dropout为0.25
        train_feed_dict = {
            input_datas: datas,
            input_labels: labels,
            dropout_placeholdr: 0.25
        }
        for step in range(50):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)

            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("训练结束，保存模型到{}".format(model_path))
    else:
        print("测试模式")
        # 如果是测试，载入参数
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # label和名称的对照关系
        label_name_dict = {
            0: "孩子",
            1: "车",
            #2: "雨伞"
        }
        # 定义输入和Label以填充容器，测试时dropout为0
        test_feed_dict = {
            input_datas: datas,
            input_labels: labels,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # 真实label与模型预测label
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
            # 将label id转换为label名
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))











