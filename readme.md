# CMP

## Deepmind_lab

### 拉取子仓库(deepmind_lab)

```bash
# more about submodule see: https://github.com/dongniu0927/toolkits
git submodule update --init --recursive
```

### python包安装

```bash
# in lab dir
bazel build -c opt python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py2-none-any.whl --force-reinstall
```

### 运行

```bash
# python
python test_deepmind_lab.py
# bazel
bazel run :game -- -l lt_chasm  # option 1
bazel run :game -- --level_script=nav_maze_random_goal_01 --level_setting=logToStdErr=true  # option 2

# levels:
[lt_chasm, lt_hallway_slope, lt_horseshoe_color, lt_space_bounce_hard, nav_maze_random_goal_01,
nav_maze_static_01, seekavoid_arena_01, stairway_to_melon]
```

### Compile maze [存疑]

```bash
bazel run :game -- -cmp_maze
recompile
```

## 参考资料
[DM lab observations](https://github.com/dongniu0927/lab/blob/master/docs/users/observations.md)  
[DM lab python api](https://github.com/dongniu0927/lab/blob/master/docs/users/python_api.md)  
[Run as human](https://github.com/dongniu0927/lab/blob/master/docs/users/run_as_human.md)  
[DM lab levels](https://github.com/dongniu0927/lab/tree/master/game_scripts/levels)