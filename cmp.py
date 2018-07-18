# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
# slim是一种轻量级的tensorflow库
from tensorflow.contrib import slim
import logging
logging.basicConfig(level=logging.DEBUG)


class CMAP(object):
    """
    论文中CMP结构的实现
    """
    def __init__(self, image_size=(959, 959), estimate_size=64, estimate_scale=3,
                 estimator=None, num_actions=7, num_iterations=12):
        """
        初始化cmp网络
        :param image_size: 输入图片的尺寸
        :param estimate_size: 估计的free space的图片的尺寸
        :param estimate_scale: 估计的规模（3通道）
        :param estimator:  自定义
        :param num_actions: 可以选择的action数目
        :param num_iterations: 循环次数
        """
        self._image_size = image_size  # 输入图片大小
        self._estimate_size = estimate_size  # 估计值规模()
        self._estimate_shape = (estimate_size, estimate_size, 3)  # 估计图片的输出尺寸,3通道
        self._estimate_scale = estimate_scale  # 多尺度
        self._num_actions = num_actions  # 可以选择的action数目
        self._num_iterations = num_iterations  # 循环次数

        # TensorFlow网络结构初始化
        self._is_training = tf.placeholder(tf.bool, name='is_training')
        self._sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length') # RNN sequence length（time steps）
        self._visual_input = tf.placeholder(tf.float32, [None, None] + list(self._image_size) + [3],
                                            name='visual_input')  # (?, ?, 84, 84, 3), 双问号代表平面坐标, 尺度代表俯视尺度大小
        self._egomotion = tf.placeholder(tf.float32, (None, None, 2), name='egomotion')
        self._reward = tf.placeholder(tf.float32, (None, None), name='reward')   #
        self._estimate_map_list = [tf.placeholder(tf.float32, (None, estimate_size, estimate_size, 3),
                                                  name='estimate_map_{}'.format(i))
                                   for i in xrange(estimate_scale)]  # 3 * shape=(?, 64, 64, 3)
        self._optimal_action = tf.placeholder(tf.float32, (None, num_actions), name='optimal_action')

        # init mapper and planner
        tensors = {}
        scaled_beliefs = self._build_mapper(tensors, estimator=estimator)
        unscaled_action = self._build_planner(scaled_beliefs, tensors)

        # get result and cal loss
        self._action = tf.nn.softmax(unscaled_action)
        self._loss = tf.losses.softmax_cross_entropy(self._optimal_action, unscaled_action)
        self._loss += tf.losses.get_regularization_loss()

        self._intermediate_tensors = tensors

    @property
    def estimate_scale(self):
        return self._estimate_scale

    @property
    def input_tensors(self):
        """
        获得网络的输入张量
        :return:
        """
        return {
            'is_training': self._is_training,
            'sequence_length': self._sequence_length,  # RNN长度
            'visual_input': self._visual_input,
            'egomotion': self._egomotion,
            'reward': self._reward,
            'estimate_map_list': self._estimate_map_list,
            'optimal_action': self._optimal_action  # 局部最优action
        }

    @property
    def intermediate_tensors(self):
        return self._intermediate_tensors

    @property
    def output_tensors(self):
        return {
            'action': self._action,
            'loss': self._loss
        }

    def _upscale_image(self, image):
        """
        将图像的scale进行调整
        :param image:
        :return:
        """
        estimate_size = self._estimate_size
        crop_size = int(estimate_size / 4)
        # print(image)  # (?, 946, 946, 2)
        # print(crop_size)  # 16
        image = image[:, crop_size:-crop_size, crop_size:-crop_size, :]
        # #第一个参数为原始图像，第二个参数为图像大小，第三个参数给出了指定的算法
        image = tf.image.resize_bilinear(image, tf.constant([estimate_size, estimate_size]),
                                         align_corners=True)
        # print(image)  # shape=(?, 64, 64, 2)
        return image

    def _build_mapper(self, m={}, estimator=None):
        """
        建立mapper结构
        :param m:
        :param estimator:
        :return:
        """
        is_training = self._is_training
        sequence_length = self._sequence_length
        visual_input = self._visual_input
        egomotion = self._egomotion
        reward = self._reward
        estimate_map_list = self._estimate_map_list
        estimate_scale = self._estimate_scale
        estimate_shape = self._estimate_shape  # 默认：(64, 64, 3)

        def _estimate(image):
            """
            对图片image的free space做估计
            :param image:
            :return:
            """
            def _xavier_init(num_in, num_out):
                stddev = np.sqrt(4. / (num_in + num_out))
                return tf.truncated_normal_initializer(stddev=stddev) # 从截断的正态分布中输出随机值（stddev为标准偏差）

            def _constrain_confidence(belief):
                """
                获得生成限制的置信度
                :param belief:
                :return:
                """
                # print(belief)  # shape=(?, 939, 939, 2), shape=(?, 64, 64, 2), shape=(?, 64, 64, 2)
                estimate, confidence = tf.unstack(belief, axis=3)  # unstack为矩阵分解
                res = tf.stack([estimate, tf.nn.sigmoid(confidence)], axis=3)  # stack为矩阵合并
                return res

            beliefs = []
            net = image  # 网络初始化时只有image输入
            # print(image)  # shape=(?, 959, 959, 3)

            # 创建mapper网络
            # arg_scope: 使得用户可以在同一个arg_scope中使用默认的参数,避免重复写函数, 参考：https://www.jianshu.com/p/c9963ae8bb2a
            # 定义3层：conv2d, fully_connected, conv2d_transpose
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                                activation_fn=tf.nn.elu,
                                biases_initializer=tf.constant_initializer(0),
                                reuse=tf.AUTO_REUSE):

                last_output_channels = 3  # 最后一层输出3通道

                # 定义第一个CNN层
                with slim.arg_scope([slim.conv2d], stride=1, padding='VALID'):
                    # 4个conv层连接到一起
                    for index, output in enumerate([(32, [7, 7]), (48, [7, 7]), (64, [5, 5]), (64, [5, 5])]):

                        channels, filter_size = output

                        # tf.nn.conv2d(input, filter, strides, padding)
                        # inputs: [batch_size, in_height, in_width, in_channels] -> [图片数量, 图片高度, 图片宽度, 图像通道数]
                        # filter: [filter_height, filter_width, in_channels, out_channels] -> [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                        # out_channels: 得到的特征图张数
                        # strides: 一维的向量，长度为4，对应的是在inputs的4个维度上的步长

                        # tf.contrib.slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME')
                        # inputs: same as top
                        # num_outputs: out_channels of filter
                        # filter_size: [卷积核的宽度，卷积核的高度]
                        # return: feature map
                        net = slim.conv2d(net, channels, filter_size, scope='mapper_conv_{}'.format(index),
                                          weights_initializer=_xavier_init(np.prod(filter_size) * last_output_channels,
                                                                           channels))
                        last_output_channels = channels

                    # print(net)  # shape=(?, 939, 939, 64)

                    # create fc layer
                    fc_output_channels = 200  # output of fc layer
                    net = slim.fully_connected(net, fc_output_channels, scope="mapper_fc",
                                               weights_initializer=_xavier_init(last_output_channels,
                                                                                fc_output_channels))
                    # print(net)  # shape=(?, 939, 939, 200)

                    last_output_channels = fc_output_channels

                # 解卷积 layer
                with slim.arg_scope([slim.conv2d_transpose],
                                    stride=1, padding='SAME'):
                    for index, output in enumerate((64, 32, 2)):  # 2代表一个estimate与一个confidence
                        # conv2d_transpose(value, filter, output_shape, strides, padding="SAME")
                        # filter: [filter_height, filter_width, out_channels, in_channels]
                        net = slim.conv2d_transpose(net, output, [7, 7], scope='mapper_deconv_{}'.format(index),
                                                    weights_initializer=_xavier_init(7 * 7 * last_output_channels,
                                                                                     output))
                        last_output_channels = output

                    # print(net)  # shape=(?, 939, 939, 2)

                    beliefs.append(net)  # 压入net网络, 原scale

                    # cal confidence
                    for i in xrange(estimate_scale - 1):
                        net = slim.conv2d_transpose(net, 2, [6, 6],
                                                    weights_initializer=_xavier_init(6 * 6 * last_output_channels, 2),
                                                    scope='mapper_upscale_{}'.format(i))
                        last_output_channels = 2
                        # print(net)  # shape=(?, 939, 939, 2)
                        beliefs.append(self._upscale_image(net))  # 处理后shape=(?, 64, 64, 2)

            return [_constrain_confidence(belief) for belief in beliefs]

        def _apply_egomotion(tensor, scale_index, ego):
            """
            CMP中W运算
            :param tensor:
            :param scale_index:
            :param ego: 包含平移与旋转信息
            :return:
            """
            translation, rotation = tf.unstack(ego, axis=1)

            cos_rot = tf.cos(rotation)
            sin_rot = tf.sin(rotation)
            zero = tf.zeros_like(rotation)
            scale = tf.constant((2 ** scale_index) / (300. / self._estimate_size), dtype=tf.float32)

            transform = tf.stack([cos_rot, sin_rot, tf.multiply(tf.negative(translation), scale),
                                  tf.negative(sin_rot), cos_rot, zero,
                                  zero, zero], axis=1)
            res = tf.contrib.image.transform(tensor, transform, interpolation='BILINEAR')
            print(res)
            return res

        def _delta_reward_map(re):
            """
            根据reward计算reward map的梯度
            :param re: reward单值
            :return:
            """
            # print("Reward size:", re.shape)
            h, w, c = estimate_shape  # 最终free space图片的高度，宽度，通道
            m_h, m_w = int((h - 1) / 2), int((w - 1) / 2)  # 中间长宽值
            # tf.pad为填充
            map = tf.pad(tf.expand_dims(re, axis=2),
                          tf.constant([[0, 0], [m_h - 1, w - m_h], [m_w - 1, w - m_w]]))
            # print(map)
            # (?, 64, 64)
            return map

        def _warp(temp_belief, prev_belief):
            """
            CMP中U函数
            :param temp_belief:
            :param prev_belief:
            :return:
            """
            # print(temp_belief)  # shape=(?, 939, 939, 2)
            # print(prev_belief)  # shape=(?, 939, 939, 3)
            temp_estimate, temp_confidence, temp_rewards = tf.unstack(temp_belief, axis=3)
            prev_estimate, prev_confidence, prev_rewards = tf.unstack(prev_belief, axis=3)

            current_confidence = temp_confidence + prev_confidence  # 更新新的置信度
            current_estimate = tf.divide(tf.multiply(temp_estimate, temp_confidence) +
                                         tf.multiply(prev_estimate, prev_confidence),
                                         current_confidence)  # 更新新的free space估计
            current_rewards = temp_rewards + prev_rewards
            current_belief = tf.stack([current_estimate, current_confidence, current_rewards], axis=3)
            return current_belief

        class BiLinearSamplingCell(tf.nn.rnn_cell.RNNCell):
            """
            RNNCell，多个cell可组成RNN
            更多RNNCell参考：https://zhuanlan.zhihu.com/p/28196873
            """
            @property
            def state_size(self):
                """
                :return: 隐层大小(隐藏单元数目)
                """
                res = [tf.TensorShape(estimate_shape)] * estimate_scale
                # logging.debug(res)
                return res

            @property
            def output_size(self):
                """
                :return: 输出大小
                """
                return self.state_size

            def __call__(self, inputs, state, scope=None):
                """
                RNNCell需要实现__call__：(output, next_state) = call(input, state)
                __call__令BiLinearSamplingCell的实例可以被自调用
                :param inputs: (图片，自运动信息，reward): shape:((?, 959, 959, 3), (?, 2), (?, 1)) --> 等价于RNN中x输入
                :param state: RNN单元状态 --> 等价于RNN中h输入
                :param scope:
                :return:
                """
                # print(image, ego, re)  # shape:((?, 959, 959, 3), shape=(?, 2), shape=(?, 1))
                # print(state)  # shape: (shape=(?, 64, 64, 3), shape=(?, 64, 64, 3), shape=(?, 64, 64, 3))
                image, ego, re = inputs
                # 计算delta reward map, tf.expand_dims在第3维上增加一维度
                delta_reward_map = tf.expand_dims(_delta_reward_map(re), axis=3)
                # print(delta_reward_map)  # shape=(?, 64, 64, 1)
                # 预测本次f_{t}
                current_scaled_estimates = _estimate(image) if estimator is None else estimator(image)
                # print(current_scaled_estimates)  # shape=(?, 939, 939, 2), shape=(?, 64, 64, 2), shape=(?, 64, 64, 2)

                # not do it
                # current_scaled_estimates = [tf.concat([estimate, delta_reward_map], axis=3)  # 拼接？？
                #                             for estimate in current_scaled_estimates]
                # print(current_scaled_estimates)

                # 获得对上一次f_{t-1}的转换
                previous_scaled_estimates = [_apply_egomotion(belief, scale_index, ego)
                                             for scale_index, belief in enumerate(state)]
                outputs = [_warp(c, p) for c, p in zip(current_scaled_estimates, previous_scaled_estimates)]
                # outputs, outputs: y1, h1
                # output其实和隐状态的值是一样的
                return outputs, outputs

        # build_mapper的返回值
        # 定义训练相关参数
        # logging.debug(visual_input)  # Tensor("visual_input:0", shape=(?, ?, 84, 84, 3), dtype=float32)
        normalized_input = slim.batch_norm(visual_input, is_training=is_training)  # 正则化输入
        # logging.debug(normalized_input)

        bi_linear_cell = BiLinearSamplingCell()
        # tf.nn.dynamic_rnn可将多个cell链接成网络
        X_in = (normalized_input, egomotion, tf.expand_dims(reward, axis=2))  # 输入
        # print(X_in)  # (shape=(?, ?, 959, 959, 3), shape=(?, ?, 2), shape=(?, ?, 1))
        # print(estimate_map_list)
        # 使用函数tf.nn.dynamic_rnn就相当于调用了n次call函数
        # inter_beliefs, final_belief: hx
        inter_beliefs, final_belief = tf.nn.dynamic_rnn(bi_linear_cell,
                                                        X_in,
                                                        sequence_length=sequence_length,  # 序列长度
                                                        dtype=tf.float32)  # TODO: initial_state
        # print(inter_beliefs, final_belief)
        m['estimate_map_list'] = inter_beliefs
        return final_belief

    def _build_planner(self, scaled_beliefs, m={}):
        is_training = self._is_training
        batch_size = tf.shape(scaled_beliefs[0])[0]
        image_scaler = self._upscale_image
        estimate_size = self._estimate_size
        value_map_size = (estimate_size, estimate_size, 1)  # 单一通道
        num_actions = self._num_actions
        num_iterations = self._num_iterations

        def _fuse_belief(belief):
            """
            混合结构（fuser）
            :param belief:
            :return:
            """
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.elu,
                                weights_initializer=tf.truncated_normal_initializer(stddev=1),
                                biases_initializer=tf.constant_initializer(0),
                                stride=1, padding='SAME', reuse=tf.AUTO_REUSE):
                # inputs: belief; outputs: 1; kernel_size: [1, 1]
                net = slim.conv2d(belief, 1, [1, 1], scope='fuser_combine')
                return net

        class HierarchicalVINCell(tf.nn.rnn_cell.RNNCell):
            """
            Value iteration module子结构Cell
            """
            @property
            def state_size(self):
                return tf.TensorShape(value_map_size)

            @property
            def output_size(self):
                return self.state_size

            def __call__(self, inputs, state, scope=None):
                # Upscale previous value map
                state = image_scaler(state)

                estimate, _, values = [tf.expand_dims(layer, axis=3)
                                       for layer in tf.unstack(inputs, axis=3)]
                with slim.arg_scope([slim.conv2d], reuse=tf.AUTO_REUSE):
                    rewards_map = _fuse_belief(tf.concat([estimate, values, state], axis=3))
                    actions_map = slim.conv2d(rewards_map, num_actions, [3, 3],
                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.42),
                                              biases_initializer=tf.constant_initializer(0),
                                              scope='VIN_actions_initial')
                    values_map = tf.reduce_max(actions_map, axis=3, keepdims=True)

                with slim.arg_scope([slim.conv2d], reuse=tf.AUTO_REUSE):
                    for i in xrange(num_iterations - 1):
                        rv = tf.concat([rewards_map, values_map], axis=3)
                        actions_map = slim.conv2d(rv, num_actions, [3, 3],
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.42),
                                                  biases_initializer=tf.constant_initializer(0),
                                                  scope='VIN_actions')
                        values_map = tf.reduce_max(actions_map, axis=3, keepdims=True)

                return values_map, values_map

        beliefs = tf.stack([slim.batch_norm(belief, is_training=is_training) for belief in scaled_beliefs], axis=1)
        vin_cell = HierarchicalVINCell()
        interm_values_map, final_values_map = tf.nn.dynamic_rnn(vin_cell, beliefs,
                                                                initial_state=vin_cell.zero_state(batch_size,
                                                                                                  tf.float32),
                                                                swap_memory=True)
        m['value_map'] = interm_values_map

        values_features = slim.flatten(final_values_map)
        actions_logit = slim.fully_connected(values_features, num_actions ** 2,
                                             weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                                             biases_initializer=tf.constant_initializer(0),
                                             activation_fn=tf.nn.elu,
                                             scope='logit_output_1')
        actions_logit = slim.fully_connected(actions_logit, num_actions,
                                             weights_initializer=tf.truncated_normal_initializer(stddev=0.5),
                                             biases_initializer=tf.constant_initializer(1.0 / num_actions),
                                             scope='logit_output_2')

        return actions_logit


def prepare_feed_dict(tensors, data):
    """
    填充tensors
    :param tensors:
    :param data:
    :return:
    """
    feed_dict = {}
    for k, v in tensors.iteritems():
        if k not in data:
            continue

        if not isinstance(v, list):
            if isinstance(data[k], np.ndarray):
                feed_dict[v] = data[k].astype(v.dtype.as_numpy_dtype)
            else:
                feed_dict[v] = data[k]
        else:
            for t, d in zip(v, data[k]):
                feed_dict[t] = d.astype(t.dtype.as_numpy_dtype)

    return feed_dict


if __name__ == "__main__":
    net = CMAP()
