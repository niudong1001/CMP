# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from env import get_game_environment
from cmp1 import CMAP
import cv2
from expert import Expert
from copy import deepcopy


def prepare_feed_dict(tensors, data):
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

    estimate_size = 256
    estimate_scale = 3
    episode_size = 360
    
    net = CMAP(image_size=(episode_size, episode_size, 3))
    exp = Expert()
    env = get_game_environment(width=str(episode_size), height=str(episode_size))
    
    while True:

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            env.reset()

            obs = env.observations()
            obs["pose.loc"] = obs["DEBUG.POS.TRANS"]
            print("Init player loc:", obs["pose.loc"][:2])
            print("Init player node(row, col):", exp.player_node(obs))
            print("Red home:", obs["DEBUG.FLAGS.RED_HOME"])
            obs["pose.angle"] = obs["DEBUG.POS.ROT"]

            # 显示
            img_tb = obs["DEBUG.CAMERA.TOP_DOWN"]
            img = obs["RGB_INTERLEAVED"]
            img_tb_rgb = []
            for i in range(episode_size):
                for j in range(episode_size):
                    d = (img_tb[0][i][j], img_tb[1][i][j], img_tb[2][i][j])
                    img_tb_rgb.append(d)
            img_tb_rgb = np.array(img_tb_rgb).reshape(episode_size, episode_size, 3)
            img = np.array(img)
            # print(img_tb_rgb.shape)
            cv2.imshow("Game", img)
            cv2.imshow("TB", img_tb_rgb)
            cv2.imwrite("init_player_view.png", img)
            cv2.imwrite("init_top_view.png", img_tb_rgb)
            cv2.waitKey(0)
            # print(obs["pose"])
            
            # init
            episode = dict()
            episode['act'] = [np.argmax(exp.get_optimal_action(obs))]
            episode['obs'] = [obs]
            episode['ego'] = [[0., 0., 0.]]
            episode['est'] = [exp.get_free_space_map(obs, estimate_size=estimate_size)]
            episode['gol'] = [exp.get_goal_map(obs, estimate_size=estimate_size)]
            episode['rwd'] = [0.]
            estimate_map_list = [np.zeros((1, estimate_size, estimate_size, 3))
                                            for _ in xrange(estimate_scale)]
            old_estimate_map_list = estimate_map_list

            # episode循环
            for _ in xrange(episode_size):
                
                prev_obs = deepcopy(episode['obs'][-1])
                optimal_action = exp.get_optimal_action(prev_obs)
                expand_dim = lambda x: np.array([[x[-1]]])

                imgd = prev_obs["RGBD_INTERLEAVED"]
                imgd = imgd[np.newaxis, np.newaxis, :]
                img = prev_obs["RGB_INTERLEAVED"]
                # print(imgd.shape)
                cv2.imshow("Game", img)
                cv2.waitKey(0)

                feed_data = {'sequence_length': np.array([1]),
                                        'visual_input': expand_dim(episode['obs']),
                                        'egomotion': expand_dim(episode['ego']),
                                        'reward': expand_dim(episode['rwd']),
                                        'space_map': expand_dim(episode['est']),
                                        'goal_map': expand_dim(episode['gol']),
                                        'estimate_map_list': estimate_map_list,
                                        'optimal_action': expand_dim(episode['act']),
                                        'optimal_estimate': expand_dim(episode['est']),
                                        'is_training': False}
                
                feed_dict = prepare_feed_dict(net.input_tensors, feed_data)

                results = sess.run([net.output_tensors['action']] +
                                    net.intermediate_tensors['estimate_map_list'], feed_dict=feed_dict)
                
                predict_action = np.argmax(np.squeeze(results[0]))
                old_estimate_map_list = estimate_map_list
                estimate_map_list = [m[0] for m in results[1:]]

                reward = env.step(predict_action)
                obs = env.observations()
                obs["pose.loc"] = obs["DEBUG.POS.TRANS"]
                obs["pose.angle"] = obs["DEBUG.POS.ROT"]

                # push data
                episode['act'].append(np.argmax(optimal_action))
                episode['obs'].append(obs)
                episode['ego'].append(env.calculate_egomotion(prev_obs, obs))
                episode['est'] = [exp.get_free_space_map(obs, estimate_size=estimate_size)]
                episode['gol'] = [exp.get_goal_map(obs, estimate_size=estimate_size)]
                episode['rwd'].append(deepcopy(reward))