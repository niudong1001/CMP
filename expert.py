# -*- coding:utf-8 -*-
from itertools import product
import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import shift, rotate
import networkx as nx
from top_view_renderer import EntityMap
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  

class Expert(object):
    def _build_free_space_estimate(self):
        # map can be find in: deepmind_lab/game_scripts/levels/nav_maze_random_goal_01.lua
        entity_map = EntityMap("./maps/cmp_maze.entityLayers")
        # print(entity_map.wall_coordinates_from_string)
        wall_coordinates = frozenset((entity_map.height() - inv_row - 1, col)
                                     for col, inv_row in entity_map.wall_coordinates_from_string((1, 1)))
        self._walls = wall_coordinates
        # print(self._walls)

        self._height = entity_map.height()
        self._width = entity_map.width()

        print("Map size:", self._width, self._height)

        self._graph.clear()
        # 将非wall的node加入图中
        self._graph.add_nodes_from((row, col)
                                   for row in xrange(entity_map.height())
                                   for col in xrange(entity_map.width())
                                   if (row, col) not in wall_coordinates)
        # draw wall
        # entity中原点为top-left
        fig = plt.figure()  
        ax1 = fig.add_subplot(111)  
        for wall in self._walls:
            # print(wall)
            ax1.scatter(wall[0], wall[1], c = 'r', marker = 'o') 
        for node in self._graph.nodes:
            ax1.scatter(node[0], node[1], c = "gray", marker = "*")
        plt.savefig('map_walls.png')
        plt.show()

        # print(self._graph.nodes)
        for row in xrange(entity_map.height()):
            for col in xrange(entity_map.width()):
                if not self._graph.has_node((row, col)):
                    continue

                # self connect
                self._graph.add_edge((row, col), (row, col), weight=0)

                # left = bottom = right = False

                # Left
                left_col = col - 1
                while self._graph.has_node((row, left_col)):
                    # left = True
                    self._graph.add_edge((row, left_col), (row, col), weight=(col - left_col) * 100)
                    left_col -= 1

                # Right
                right_col = col + 1
                while self._graph.has_node((row, right_col)):
                    # right = True
                    self._graph.add_edge((row, right_col), (row, col), weight=(right_col - col) * 100)
                    right_col += 1

                # Bottom
                bottom_row = row + 1
                while self._graph.has_node((bottom_row, col)):
                    # bottom = True
                    self._graph.add_edge((bottom_row, col), (row, col), weight=(bottom_row - row) * 100)
                    bottom_row += 1

                # Bottom
                top_row = row - 1
                while self._graph.has_node((top_row, col)):
                    # bottom = True
                    self._graph.add_edge((top_row, col), (row, col), weight=(row - top_row) * 100)
                    top_row -= 1

                # Bottom-Left
                bottom_row = row + 1
                left_col = col - 1
                if self._graph.has_node((bottom_row, left_col)):
                    weight = int(np.sqrt(2) * (bottom_row - row) * 100)
                    self._graph.add_edge((bottom_row, left_col), (row, col), weight=weight)

                # Bottom-Right
                bottom_row = row + 1
                right_col = col + 1
                if self._graph.has_node((bottom_row, right_col)):
                    weight = int(np.sqrt(2) * (bottom_row - row) * 100)
                    self._graph.add_edge((bottom_row, right_col), (row, col), weight=weight)

        self._weights = dict(nx.shortest_path_length(self._graph, weight='weight'))
        # print(self._weights)

    def player_node(self, obs):
        return self._player_node(obs)

    def _player_node(self, obs):
        loc = obs.get('pose.loc')[:2]
        # print(self._width, self._height)
        x, y = loc
        # x, y = int(loc[0] / 1000 * self._width), int(loc[1] / 500 * self._height)
        # print(x, y)
        return self._get_rowcol(x, y)
    
    def _get_rowcol(self, x, y):
        """
        根据x,y坐标得出所在行与列,行列原点为entityLayers的左上
        """
        row, col = int((500 - y) / 500 * self._height), int( x / 1000 * self._width)
        return row, col

    def _goal_node(self, obs):
        goal = obs.get('pose.loc')[:2]
        x, y = goal
        return self._get_rowcol(x, y)

    def _node_to_game_coordinate(self, node):
        row, col = node
        return (col + 0.5) * 1000, (self._height - row - 0.5) * 500

    def __init__(self):
        self._graph = nx.Graph()
        self._weights = {}
        self._build_free_space_estimate()

    def get_goal_map(self, obs, game_size=1280, estimate_size=256):
        goal_map = np.zeros((estimate_size, estimate_size))
        game_scale = 1 / (game_size / float(estimate_size))
        block_scale = int(100 * game_scale / 2)

        player_pos, player_rot = obs.get('pose.loc')[:2], obs.get('pose.angle')[1]
        goal_pos = np.array(self._node_to_game_coordinate(self._goal_node(obs)))
        delta_pos = (goal_pos - player_pos) * game_scale
        # delta_angle = np.arctan2(delta_pos[1], delta_pos[0]) - player_rot

        c, s = np.cos(player_rot), np.sin(player_rot)
        rot_mat = np.array([[c, s], [-s, c]])
        x, y = np.dot(rot_mat, delta_pos).astype(np.int32)
        w = int(estimate_size / 2) + x
        h = int(estimate_size / 2) - y

        goal_map[h - block_scale:h + block_scale, w - block_scale:w + block_scale] = 1

        return np.expand_dims(goal_map, axis=2)

    def get_free_space_map(self, obs, game_size=1280, estimate_size=256):
        image = np.zeros((estimate_size * 2, estimate_size * 2), dtype=np.uint8)
        game_scale = 1 / (game_size / float(estimate_size))
        block_scale = 100 * game_scale

        for row, col in product(xrange(self._height), xrange(self._width)):
            if (row, col) in self._walls:
                continue
            w = int(col * block_scale)
            h = int((row - self._height) * block_scale)
            size = int(block_scale)
            w_end = w + size
            h_end = h + size if (h + size) != 0 else estimate_size * 2
            image[h: h_end, w: w_end] = 255

        player_pos, player_rot = obs['pose.loc'][:2], obs['pose.angle'][1]
        w, h = player_pos * game_scale

        w -= estimate_size
        h -= estimate_size

        image = shift(image, [h, -w])
        image = rotate(image, -1 * np.rad2deg(player_rot))

        h, _ = image.shape
        crop_size = int((h - estimate_size) / 2)
        if crop_size > 0:
            image = image[crop_size:-crop_size, crop_size:-crop_size]

        image = imresize(image, size=(estimate_size, estimate_size))
        assert image.shape[0] == estimate_size

        return np.expand_dims(image, axis=2)

    def get_optimal_action(self, obs):
        player_x, player_y, player_angle = obs.get('pose.loc')[0], obs.get('pose.loc')[1], obs.get('pose.angle')[1]

        goal_node = self._goal_node(obs)

        get_norm_angle = lambda angle: np.arctan2(np.sin(angle), np.cos(angle))
        get_game_angle = lambda x, y: np.arctan2(y - player_y, x - player_x)
        get_node_angle = lambda node: get_game_angle(*self._node_to_game_coordinate(node))
        node_criterion = lambda node: self._weights[node][goal_node] + \
                                      get_norm_angle(get_node_angle(node) - player_angle) / np.pi

        optimal_node = min(self._graph.neighbors(self._player_node(obs)), key=node_criterion)

        action = np.zeros(4)

        if self._player_node(obs) == goal_node:
            action[2] = 1
            return action

        angle_delta = get_norm_angle(get_node_angle(optimal_node) - player_angle)

        if abs(angle_delta) < np.deg2rad(7.5):
            action[2] = 1
        else:
            if angle_delta < 0:
                action[0] = 1
            else:
                action[1] = 1

        return action
