# UAV 三维深度强化学习轨迹规划

本项目第二阶段使用 DQN 实现带点质量动力学的三维无人机轨迹规划。输入三维起点和目标点后，智能体会在建筑物、地图边界、水平/三维速度、升降速度、爬升角、加减速度、急动度和终点速度约束下学习飞向目标点。

状态包含位置、目标相对量、速度、上一时刻实际加速度、速度/加速度占比、制动裕度、26 向雷达和时间进度。27 个离散动作表示 26 个方向的正常飞行加速度指令与一个零加速度滑行动作。环境使用固定时间步直接积分得到轨迹点，因此输出的速度和加速度是 RL 决策过程的一部分，而非事后轨迹平滑结果。

## 项目结构

```text
uav_drl_path_planning.py       # 主函数和命令行的入口
uav_drl/
  actions.py                   # 三维加速度动作，26 个方向 + coast
  config.py                    # 三维地图、建筑物、环境参数
  scenes.py                    # 四地图注册表和住宅区障碍物适配器
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
python .\uav_drl_path_planning.py --scene wujing_airfield --episodes 3 --eval-episodes 1 --no-plots
```

## 分别训练四张地图

可先查看场景名称：

```powershell
python .\uav_drl_path_planning.py --list-scenes
```

四张地图使用同一个训练入口，通过 `--scene` 选择：

```powershell
python .\uav_drl_path_planning.py --scene wujing_airfield --episodes 1500 --eval-episodes 20 --visualize
python .\uav_drl_path_planning.py --scene lanxianghu_villa --episodes 1500 --eval-episodes 20 --visualize
python .\uav_drl_path_planning.py --scene sanming_garden --episodes 1500 --eval-episodes 20 --visualize
python .\uav_drl_path_planning.py --scene spring_garden_phase2 --episodes 1500 --eval-episodes 20 --visualize
```

每个场景只会自动加载自己的 checkpoint。希望丢弃该场景旧模型并从头训练时，
增加 `--fresh-start`。三个住宅区的旋转矩形建筑会保守转换成轴对齐包围盒，
默认外扩 2 m；吴泾试飞场仍使用 8 m 外扩量。可用 `--obstacle-inflation`
覆盖当前场景的默认值。

默认动力学参数来自飞行日志：最大水平速度 23.0 m/s、最大三维合速度
23.18 m/s、最大上升速度 2.85 m/s、最大下降速度 1.65 m/s、最大爬升角
90°、瞬时峰值加速度 15.5 m/s²、最大减速度 3.09 m/s²。正常飞行加速度
采用给定 3--5 m/s² 区间的保守端 3.0 m/s²；Jerk 约束采用平滑后峰值
78 m/s³，并保留原始日志峰值 142 m/s³ 作为参考。上述参数均可通过对应命令行参数覆盖。

## 轨迹形状奖励

当前奖励版本为 v2，除了目标进度、碰撞和动力学奖励外，还包括：

- `extra_altitude`：只惩罚高于起终点参考高度走廊和局部建筑安全越障高度的部分；
- `detour`：惩罚航段长度中没有转化为目标进度的绕行距离；
- `turn`：惩罚相邻速度方向的转角；
- `goal_altitude`：接近目标时逐渐增强目标高度误差惩罚；
- `vertical_speed_guidance`：引导垂直速度跟踪由剩余高度误差计算出的期望值，并在目标高度附近收敛到零。

默认参数可以通过以下选项调整：

```text
--extra-altitude-penalty-scale 0.12
--extra-altitude-margin 3.0
--detour-penalty-scale 0.35
--turn-penalty-scale 0.20
--goal-guidance-distance 60.0
--goal-altitude-penalty-scale 0.08
--vertical-speed-guidance-scale 0.30
--vertical-guidance-time 4.0
```

每一步 `info["reward_components"]` 会记录各奖励分项，
`info["reward_diagnostics"]` 会记录参考高度、额外高度、转角、绕路距离和期望
垂直速度。v1 checkpoint 的经验回放包含旧奖励，升级后必须使用
`--fresh-start` 训练新模型。

## 指定三维起点和目标点

```powershell
python .\uav_drl_path_planning.py --episodes 1500 --fresh-start --start-x 20 --start-y 100 --start-z 8 --target-x 480 --target-y 260 --target-z 12 --visualize
```

## 加载模型进行规划

```powershell
python .\uav_drl_path_planning.py --scene wujing_airfield --skip-train --load-model .\outputs\wujing_airfield\models\wujing_airfield_dqn.pt --start-x 20 --start-y 100 --start-z 8 --target-x 480 --target-y 260 --target-z 12 --visualize
```

建筑水平外扩量可通过 `--obstacle-inflation` 调整，例如设为 10 m。更换场景后
应使用 `--fresh-start` 并重新训练，避免续载旧地图的模型和经验回放池。

## 导出并自动加载 QGroundControl 任务

QGC 导出是可选的，不会改变训练、评估或原有 CSV 输出。只有轨迹完整验证通过，
且抽稀后的直线任务航段仍然无碰撞时，程序才会写入 Plan 文件。

先在 QGC 的 **Settings > General** 中完成一次性设置：

1. 启用 `Autoload Missions`；
2. 记下 `Application Load/Save Path`；
3. 确认车辆的 MAVLink System ID（通常为 1）。

然后显式提供本地 ENU `(0, 0, 0)` 对应的实测 WGS-84 原点和 AMSL 海拔：

```powershell
python .\uav_drl_path_planning.py `
  --scene wujing_airfield `
  --skip-train `
  --load-model .\outputs\wujing_airfield\models\wujing_airfield_dqn.pt `
  --start-x 20 --start-y 100 --start-z 8 `
  --target-x 480 --target-y 260 --target-z 12 `
  --export-qgc-plan `
  --qgc-origin-lat-wgs84 31.0000000 `
  --qgc-origin-lon-wgs84 121.0000000 `
  --qgc-origin-alt-amsl 6.0 `
  --qgc-firmware px4 `
  --qgc-system-id 1 `
  --qgc-autoload-dir "C:\QGC-LoadSave"
```

请把示例原点替换成现场测得的数据。地图配置中的原点是 GCJ-02，不能直接作为
飞控使用的 WGS-84 GPS 原点。程序会在本次运行目录保存 `qgc_mission.plan`，并将
同一计划原子更新为 `<QGC Load/Save Path>/AutoLoad1.plan`。QGC 会在车辆下一次连接
时自动加载并上传该任务；已经连接时可重新连接，或在 Plan View 中打开运行目录的
`qgc_mission.plan`。

默认计划包含 5 m 垂直起飞和后续航点，不会自动解锁或启动任务，也不会擅自添加
返航/降落。可用 `--qgc-takeoff-altitude` 修改起飞高度，用
`--qgc-end-action rtl` 或 `--qgc-end-action land` 显式指定结束动作。正式飞行前应先在
PX4/ArduPilot SITL 中检查航点、高度基准、围栏和返航行为。

## 输出文件

模型与运行结果按场景隔离：

```text
outputs/
  wujing_airfield/
    models/wujing_airfield_dqn.pt
    runs/run_时间戳/
  lanxianghu_villa/
    models/lanxianghu_villa_dqn.pt
    runs/run_时间戳/
  sanming_garden/
    models/sanming_garden_dqn.pt
    runs/run_时间戳/
  spring_garden_phase2/
    models/spring_garden_phase2_dqn.pt
    runs/run_时间戳/
```

每次运行目录中的重点文件包括：

- `run_config.json`：本次场景、模型路径和完整环境配置；
- `best_timed_trajectory.csv`：等时间轨迹点及速度、加速度；
- `trajectory_validation.json`：速度/加速度/急动度、积分一致性、碰撞、边界和目标到达验证；
- `qgc_mission.plan`：启用 `--export-qgc-plan` 后生成的 QGroundControl 任务；
- `best_trajectory.png`、`trajectory_profiles.png`：三维轨迹和动力学曲线。

该目录已加入 `.gitignore`，默认不会提交到仓库。第一阶段 checkpoint 的网络输入和动作语义不同，不能直接用于第二阶段，应重新训练。
