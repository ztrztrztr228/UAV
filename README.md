# UAV 三维深度强化学习轨迹规划

本项目第二阶段使用 DQN 实现带点质量动力学的三维无人机轨迹规划。输入三维起点和目标点后，智能体会在建筑物、地图边界、最大速度、最大加速度、最大急动度和终点速度约束下学习飞向目标点。

状态包含位置、目标相对量、速度、上一时刻实际加速度、速度/加速度占比、制动裕度、26 向雷达和时间进度。27 个离散动作表示 26 个方向的最大加速度指令与一个零加速度滑行动作。环境使用固定时间步直接积分得到轨迹点，因此输出的速度和加速度是 RL 决策过程的一部分，而非事后轨迹平滑结果。

## 项目结构

```text
uav_drl_path_planning.py       # 主函数和命令行的入口
uav_drl/
  actions.py                   # 三维加速度动作，26 个方向 + coast
  config.py                    # 三维地图、建筑物、环境参数
  environment.py               # 三维环境、状态空间、奖励函数、碰撞检测
  model.py                     # DQN 神经网络
  agent.py                     # DQN 智能体和经验回放池
  training.py                  # 训练和评估流程
  visualization.py             # 训练曲线、三维轨迹图、CSV 保存
  validation.py                # 碰撞、边界、动力学积分和终点验证
  utils.py                     # 随机种子、三维坐标解析
data/
  wujing_airfield_building_estimates.csv  # 吴泾试飞场公开影像估算数据
UAV_DRL_项目提纲.md             # 项目说明和报告提纲
```

## 吴泾试飞场估算场景

默认地图已经切换为吴泾试飞场估算场景。局部坐标原点为 GCJ-02
`(121.4480000, 31.0680000)`，`x` 向东、`y` 向北、`z` 向上，单位均为米。
当前仿真范围为 `x=[0,500]`、`y=[-30,280]`、`z=[0,50]`。

10 栋建筑按 CSV 中的中心点和轴对齐尺寸建模，默认水平外扩 8 m。
B01-B08 的高度暂假设为 15 m，B09-B10 暂假设为 25 m。这些数值来自公开
影像估算和保守假设，只适用于算法仿真，不能直接用于真实飞行。

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 快速测试

```powershell
python .\uav_drl_path_planning.py --episodes 3 --eval-episodes 1 --no-plots
```

## 训练三维模型

```powershell
python .\uav_drl_path_planning.py --episodes 1500 --eval-episodes 20 --trajectory-dt 0.5 --max-speed 8 --max-acceleration 3 --max-jerk 12 --visualize
```

## 指定三维起点和目标点

```powershell
python .\uav_drl_path_planning.py --episodes 1500 --fresh-start --start-x 20 --start-y 100 --start-z 8 --target-x 480 --target-y 260 --target-z 12 --visualize
```

## 加载模型进行规划

```powershell
python .\uav_drl_path_planning.py --skip-train --load-model .\outputs\wujing_airfield_dqn.pt --start-x 20 --start-y 100 --start-z 8 --target-x 480 --target-y 260 --target-z 12 --visualize
```

建筑水平外扩量可通过 `--obstacle-inflation` 调整，例如设为 10 m。更换场景后
应使用 `--fresh-start` 并重新训练，避免续载旧地图的模型和经验回放池。

## 输出文件

训练和评估结果默认保存在 `outputs/run_时间戳/` 下，重点文件包括：

- `best_timed_trajectory.csv`：等时间轨迹点及速度、加速度；
- `trajectory_validation.json`：速度/加速度/急动度、积分一致性、碰撞、边界和目标到达验证；
- `best_trajectory.png`、`trajectory_profiles.png`：三维轨迹和动力学曲线。

该目录已加入 `.gitignore`，默认不会提交到仓库。第一阶段 checkpoint 的网络输入和动作语义不同，不能直接用于第二阶段，应重新训练。
