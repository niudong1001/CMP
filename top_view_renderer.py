from __future__ import print_function
import os
import getpass
import logging
import heapq

import numpy as np
import matplotlib as mplib
mplib.use('Agg')
import matplotlib.cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.backends import pylab_setup
import sys
import math

def euclidean(x):
    return np.sqrt(np.sum(x**2))


def distance_to_edge(vertex_pair, pt):
    assert len(vertex_pair) == 2, "Expect a pair"
    assert vertex_pair[0].shape == (2,), "Expect 2D point {}".format(vertex_pair[0])
    normal = (vertex_pair[1] - vertex_pair[0]).dot([[0, -1],
                                                    [1,  0]])
    normal = normal / euclidean(normal)
    h = - normal.dot(vertex_pair[0])
    if h > 0:
        # to ensure that normal.dot(x) == -h = distance from origin
        # normal is away from origin and towards the plane?
        normal = - normal
        h = -h
    # Equation of edge is normal.dot(x) + h = 0  where h is -ve
    # Signed distance from plane that is +ve when away from origin
    proj_dist = normal.dot(pt) + h
    proj_pt = pt - proj_dist * normal
    # The vectors have to change direction 
    costheta = (proj_pt - vertex_pair[0]).dot(proj_pt - vertex_pair[1])
    # assert np.allclose(costheta, 0) or np.allclose(costheta, 2*np.pi), \
    #     "The proj_pt is on line joining vertex_pair. costheta {}".format(
    #         costheta)
    if costheta > 0:
        # projected point is outside the vertex_pair
        # print("Computing distance from points {}".format(vertex_pair))
        return min(euclidean(pt-x) for x in vertex_pair)
    else:
        # projected point is within the vertex pair
        # print("Computing distance from edge {}".format(vertex_pair))
        return abs(proj_dist)


class DistanceTransform(object):
    def __init__(self):
        self._entity_map = None
        self._wall_left_bottom = None

    def set_wall_coordinates(self, wall_left_bottom):
        # Each point is a numpy array
        self._wall_left_bottom = wall_left_bottom

    @staticmethod
    def four_corners_of_wall(mid, block_size):
        four_corners = (  mid + [0, 0]
                          , mid + [block_size[0], 0]
                          , mid + [0, block_size[1]]
                          , mid + block_size )
        return four_corners

    def distance(self, point, block_size):
        top_n = 4
        closest_wall_left_bottom = heapq.nsmallest(
            top_n
            , self._wall_left_bottom
            , key = lambda x : euclidean(point - (x + block_size/2))
        )
        # Each := (v1, v2)
        closed_edges = [
            heapq.nsmallest(
                2
                , self.four_corners_of_wall(np.asarray(w), block_size)
                , key = lambda x : euclidean(point - x))
            for w in closest_wall_left_bottom]
        return min( distance_to_edge(e, point) for e in closed_edges )


class EntityMap(object):
    def __init__(self, entity_layer_file
                 , distance_transform=DistanceTransform()):
        self.entity_layer_file = entity_layer_file
        self._entity_layer_lines = None
        self._width = None
        self._distance_transform = distance_transform
        self._wall_coordinates = None
        self._wall_coordinates_block_size = None

    def entity_layer_lines(self):
        if self._entity_layer_lines is None:
            with open(self.entity_layer_file) as ef:
                self._entity_layer_lines = ef.readlines()
        return self._entity_layer_lines

    def _wall_coordinates_from_string(self, size, wallchar="*"):
        wall_coords = []
        for row, line in enumerate(self.entity_layer_lines()):
            # print("row:", str(row))
            row_inv = self.height() - row - 1
            for col, char in enumerate(line):
                # print("col:", str(col), " value:", char)
                if char == wallchar:
                    coord = (col * size[0], row_inv * size[1])
                    # print(coord)
                    yield coord

    def goal_coordinates_from_string(self, size=[1, 1]):
        return sum([[(r+1, c+1)
                     for c, char in enumerate(line) if char == "G"]
                    for r, line in enumerate(self.entity_layer_lines())], [])

    def spawn_coordinates_from_string(self, size=[1, 1]):
        return sum([[(r+1, c+1)
                     for c, char in enumerate(line) if char == "P"]
                    for r, line in enumerate(self.entity_layer_lines())], [])

    def wall_coordinates_from_string(self, size=np.asarray((100, 100))):
        if not self._wall_coordinates:
            self._wall_coordinates_block_size = size
            self._wall_coordinates = list(
                self._wall_coordinates_from_string(size))
        assert np.all(self._wall_coordinates_block_size == size), \
            "But you said block_size = {} earlier, ".format(self._wall_coordinates_block_size) \
            + "Now you are saying {}".format(size) \
            + "I don't know what to do; you indecisive caller!!"
        return self._wall_coordinates

    def height(self):
        return len(self.entity_layer_lines())

    def width(self):
        if self._width is None:
            self._width = max(len(l) for l in self.entity_layer_lines()) - 1
        return self._width

    def distance(self, point, block_size):
        self._distance_transform.set_wall_coordinates(
            self.wall_coordinates_from_string(block_size) )
        return self._distance_transform.distance(point, block_size)


class MatplotlibVisualizer(object):
    def __init__(self):
        self._render_fig_manager = None
        self._render_backend_mod = None

    def render(self, fig):
        if self._render_fig_manager is None:
            # Chooses the backend_mod based on matplotlib configuration
            mplib.interactive(True)
            self._render_backend_mod = pylab_setup()[0]
            self._render_fig_manager = \
                self._render_backend_mod.new_figure_manager_given_figure(1, fig)
        self._render_fig_manager.canvas.figure = fig
        self._render_fig_manager.canvas.draw()
        self._render_backend_mod.show(block=False)
        return self._render_fig_manager

    def print_figure(self, fig, filename, dpi):
        #FigureCanvasAgg(fig).print_figure(filename, dpi=dpi)
        fig.set_canvas(FigureCanvasPdf(fig))
        fig.savefig(filename)

class TopView(object):
    def __init__(self, assets_top_dir=None, level_script=None, draw_fq=10, method="3D"):
        self._ax = None
        self.draw_fq = draw_fq
        self.assets_top_dir = assets_top_dir
        self.level_script = level_script
        self.block_size = np.asarray((100, 100))
        self._entity_map = EntityMap(self._entity_file())
        self.method = method
        self._top_view_episode_map = TopViewEpisodeMap(self)
        self._mplib_visualizer = MatplotlibVisualizer()

    def set_entity_layer(self, entity_layer):
        self.level_script = entity_layer

    def distance(self, point):
        return self._entity_map.distance(point, self.block_size)

    def render(self, fig):
        self._mplib_visualizer.render(fig)

    def print_figure(self, fig, filename, dpi):
        self._mplib_visualizer.print_figure(fig, filename, dpi)

    def supported(self):
        return os.path.exists(self._entity_file())

    def _make_axes(self):
        fig = mplib.figure.Figure(figsize=(4, 4))
        # TODO: axes width/height are assumed to be in 1:1 ratio
        # and expected to be handled in by set_aspect later. This is
        # problematic because we depend on matplotlib magic
        ax = fig.gca() if fig.axes else fig.add_axes([0, 0, 1, 1])
        return ax

    def get_axes(self):
        if self._ax is None:
            self._ax = self._make_axes()
        return self._ax

    def _entity_file(self):
        maze_parts = self.level_script.split('-')
        folder = maze_parts[0]
        maze_no = maze_parts[-1]
        return os.path.join(
            self.assets_top_dir
            , "assets/entityLayers/09x09/{}/entityLayers/{}.entityLayer".format(folder, maze_no))

    def add_pose(self, pose, reward=0):
        if self.supported():
            self._top_view_episode_map.add_pose(pose, reward=reward)

    def add_goal(self, goal_loc):
        if self.supported():
            self._top_view_episode_map.add_goal(goal_loc)

    def draw(self):
        if self.supported():
            self._top_view_episode_map.draw()
            return self.get_axes().figure

    def reset(self):
        if self.supported():
            self._top_view_episode_map = TopViewEpisodeMap(self)

    def draw_spawn(self, ax, spawn_loc):
        self._top_view_episode_map.draw_spawn(ax, spawn_loc)


class TopViewEpisodeMap(object):
    def __init__(self, top_view):
        self._top_view = top_view
        self._entity_map = top_view._entity_map
        self.poses2D = np.empty((0,3)) # x,y,yaw
        self.rewards = []
        self._goal_loc = None
        self._drawn_once = False
        self._added_goal_patch = None
        self._added_scatter = None
        self._added_arrow = None
        self._added_circle = None
        self.method = top_view.method

    def add_pose(self, pose, reward=0):
        self.poses2D = np.vstack((self.poses2D, (pose[0], pose[1], pose[4])))
        self.rewards.append(reward)

    def add_goal(self, goal_loc):
        self._goal_loc = goal_loc

    def last_pose(self):
        return self.poses2D[-1, :]

    def draw(self):
        if self.poses2D.shape[0] == 1 \
           or self.poses2D.shape[0] % self._top_view.draw_fq == 0:
            self._draw()

    def map_height(self):
        return self._entity_map.height()

    def map_width(self):
        return self._entity_map.width()

    def wall_coordinates_from_string(self, **kwargs):
        return self._entity_map.wall_coordinates_from_string(**kwargs)

    @property
    def block_size(self):
        return self._top_view.block_size

    def get_axes(self):
        return self._top_view.get_axes()

    def _goal_patch(self, coord, color='g', fill=True):
        goal_size = self.block_size * 0.67
        goal_pos_offset = (self.block_size - goal_size) / 2
        return mplib.patches.Rectangle(
            coord+goal_pos_offset,
            goal_size[0], goal_size[1] , color=color, fill=fill)
    
    def _text_patch(self, coord):
        goal_size = self.block_size * 0.67
        goal_pos_offset = (self.block_size - goal_size) / 2
        
        return mplib.patches.Rectangle( coord+goal_pos_offset,
            goal_size[0], goal_size[1] , color='g' , fill=True)

    def draw_spawn(self, ax, spawn_loc):
        xyblocks = np.asarray(
            (spawn_loc[1] - 1, self.map_height() - spawn_loc[0]))
        xy = xyblocks * self.block_size
        ax.add_patch(self._goal_patch(xy, color='y'))
        ax.text(xy[0]+35, xy[1]+35, "S", fontsize=10)

    def _draw_goal(self, ax):
        goal_loc = self._goal_loc
        xyblocks = np.asarray((goal_loc[1] - 1, self.map_height() - goal_loc[0]))
        xy = xyblocks * self.block_size
        if self._added_goal_patch:
            self._added_goal_patch.remove()
        self._added_goal_patch = ax.add_patch(self._goal_patch(xy))
        l = self._added_goal_patch.xy
        ax.text(l[0]+25, l[1]+20, "G", fontsize=10)

    def _wall_patch(self, coord):
        return mplib.patches.Rectangle(
                coord, self.block_size[0], self.block_size[1]
                , fill=True, facecolor='blue')

    def _draw_map(self, ax):
        for coord in self.wall_coordinates_from_string(size=self.block_size):
            ax.add_patch(self._wall_patch(coord))

    def normalized_rewards(self):
        rew = np.asarray(self.rewards)
        min_, max_ =  np.min(rew), min(0.5, np.max(rew))
        if max_ != min_:
            rew = (rew - min_) / (max_ - min_)
        return rew

    def _draw(self):
        self._draw_once()
        ax = self.get_axes()
        if self._goal_loc is not None:
            self._draw_goal(ax)
        
        if self.method != "2D":
            if self._added_scatter:
                self._added_scatter.remove()

            if self.poses2D.shape[0]:
                self._added_scatter = ax.scatter(
                    self.poses2D[:, 0], self.poses2D[:, 1]
                    , c=self.normalized_rewards()[:]
                    , cmap='coolwarm'
                    , linewidths=0
                    , edgecolors=None
                    , marker='.')
        
            if self._added_arrow:
                self._added_arrow.remove()
            
            if self.poses2D.shape[0]:
                _x   = self.poses2D[-1, 0]
                _y   = self.poses2D[-1, 1]
                _yaw = self.poses2D[-1, 2]
                _dx = 15*math.cos(_yaw)
                _dy = 15*math.sin(_yaw)
                self._added_arrow = \
                        ax.arrow(_x, _y, _dx, _dy, head_width=15, 
                                 head_length=15, fc='k', ec='k')
        else:
            # View used as input for gridworld
            
            if self._added_arrow:
                self._added_arrow.remove()
            
            if self._added_circle:
                self._added_circle.remove()
            
            if self.poses2D.shape[0]:
                _x   = self.poses2D[-1, 0]
                _y   = self.poses2D[-1, 1]
                _yaw = self.poses2D[-1, 2]
                _dx = .15*math.cos(_yaw)
                _dy = .15*math.sin(_yaw)
                self._added_arrow = \
                        ax.arrow(_x, _y, _dx, _dy, 
                                 head_width=50, head_length=50, fc='k', ec='k')
                               

                self._circle = mplib.patches.Circle((_x, _y), radius=25, color='k')
                self._added_circle = ax.add_patch(self._circle)
   
    def _draw_once(self):
        if not self._drawn_once:
            ax = self._top_view.get_axes()
            ax.clear()
            # Do not use ax.axis('equal') because it sets adjustable='datalim'
            # which cases xlim/ylim to change later.
            ax.set_aspect('equal', adjustable='box')
            ax.set_autoscale_on(False)
            ax.autoscale_view(tight=False)
            ax.set_xlim(0, self.map_width() * self.block_size[0])
            ax.set_ylim(0, self.map_height() * self.block_size[1])
            ax.set_xticks([])
            ax.set_yticks([])
            self._draw_map(ax)
            self._drawn_once = True


def render_entity_layer(entityLayer):
    # Test top view rendering
    level_script = os.path.basename(entityLayer)
    assets_top_dir = os.path.join(
        os.path.dirname(__file__ or "."), "..")
    top_view = TopView(assets_top_dir, "xyz")
    top_view._entity_file = lambda : entityLayer
    top_view._entity_map = EntityMap(top_view._entity_file())
    assert top_view.supported(), "should be supported"
    top_view.reset()
    #for _ in range(100):
    #    pose = np.random.rand(6) * top_view._entity_map.height() * top_view.block_size[1]
    #    # print("Adding pose {}".format(pose)) 
    #    #top_view.add_pose(pose) 
    #    #goal_loc = [[2, 3],[3, 2], [6, 8], [8,6]][np.random.choice(4)]
    #    # print("Adding goal at {}".format(goal_loc))
    #    #top_view.add_goal(goal_loc) 
    goal_loc = top_view._entity_map.goal_coordinates_from_string()[0]
    print("goal_loc : {}".format(goal_loc))
    top_view.add_goal(goal_loc)
    spawn_loc = top_view._entity_map.spawn_coordinates_from_string()[0]
    print("spawn_loc: {}".format(spawn_loc))
    fig = top_view.draw()
    top_view.draw_spawn(top_view.get_axes(), spawn_loc)
    #top_view.render(fig)
    filename = "/tmp/{}_{}_entityLayer.pdf".format(getpass.getuser(), level_script)
    print("Writing figure to file {}".format(filename))
    top_view.print_figure(fig , filename , dpi=80)

    
if __name__ == '__main__':
    render_entity_layer("/tmp/nav_maze_static_01.entityLayer")
    render_entity_layer("/tmp/nav_maze_static_02.entityLayer")
