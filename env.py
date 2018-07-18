# -*- coding:utf-8 -*-
import deepmind_lab
import numpy as np


def get_game_environment(level="nav_maze_random_goal_01", width=160, height=160):
    """
    get deepmind_lab env (see:https://github.com/dongniu0927/lab/blob/master/docs/users/observations.md)
    :param level:
    :return:
    """
    # RGB_INTERLEAVED: Player view interleaved
    # RGBD_INTERLEAVED: Player view with depth
    # DEBUG.CAMERA.TOP_DOWN: top view
    # DEBUG.POS.TRANS: Player's world position
    # DEBUG.POS.ROT: orientation in degrees
    # DEBUG.FLAGS.RED_HOME: Red flag's home location and state
    obses = ["RGB_INTERLEAVED", "RGBD_INTERLEAVED", "DEBUG.POS.ROT", 
            "DEBUG.CAMERA.TOP_DOWN", "DEBUG.POS.TRANS", "DEBUG.PLAYERS.TEAM", 
            "DEBUG.PLAYERS.EYE.ROT", "DEBUG.FLAGS.RED_HOME", "DEBUG.PLAYERS.EYE.POS",
            "DEBUG.FLAGS.RED", "DEBUG.FLAGS.BLUE", "DEBUG.FLAGS.BLUE_HOME", "DEBUG.FLAGS.RED_HOME"]
    lab = deepmind_lab.Lab(level, obses,
                       {'fps': '30', 'width': width, 'height': height, 
                       "maxAltCameraWidth": width, "maxAltCameraHeight": height})
    lab.reset(seed=1)
    return lab


def calculate_egomotion(previous_pose, current_pose):
    """
    计算自运动信息
    :param previous_pose:
    :param current_pose:
    :return:
    """
    previous_pos, previous_angle = previous_pose[:2], previous_pose[4]
    current_pos, current_angle = current_pose[:2], current_pose[4]

    rotation = current_angle - previous_angle
    abs_translation = current_pos - previous_pos
    abs_angle = np.arctan2(abs_translation[1], abs_translation[0])
    delta_angle = abs_angle - current_angle
    translation = np.array([np.cos(delta_angle), np.sin(delta_angle)]) * np.linalg.norm(abs_translation)

    return translation.tolist() + [rotation]
