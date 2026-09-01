# UAV 三维深度强化学习轨迹规划

本项目第二阶段使用 DQN 实现带点质量动力学的三维无人机轨迹规划。输入三维起点和目标点后，智能体会在建筑物、地图边界、水平/三维速度、升降速度、爬升角、加减速度、急动度和终点速度约束下学习飞向目标点。

状态包含位置、目标相对量、速度、上一时刻实际加速度、速度/加速度占比、制动裕度、26 向雷达、时间进度，以及停车距离、前向雷达、制动裕量和预计碰撞时间 4 个提前制动量。27 个离散动作表示 26 个方向的正常飞行加速度指令与一个零加速度滑行动作。环境使用固定时间步直接积分得到轨迹点，因此输出的速度和加速度是 RL 决策过程的一部分，而非事后轨迹平滑结果。

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

### 导出到 Gazebo

下面的命令会生成可直接启动的自包含 SDF world，以及可插入其他 world 的模型包：

```powershell
python -m gazebo_maps.export_wujing_gazebo
```

Gazebo Sim 使用 `gz sim gazebo/wujing_airfield/worlds/wujing_airfield.sdf` 启动。
Gazebo Classic 11 请增加 `--sdf-version 1.7` 重新生成。详细的模型搜索路径配置和
坐标注意事项见 `gazebo_maps/README.md`。默认导出原始建筑尺寸；如需把训练安全区
做成实体碰撞体，可增加 `--inflation 8`。

## 安装依赖

当前 `requirements.txt` 使用 CUDA 12.6 版 PyTorch，适用于本机 NVIDIA RTX 4060：

```powershell
python -m pip install -r requirements.txt
```

训练默认使用 CPU。只有明确指定 `--device cuda` 或 `--device auto` 时才会使用 GPU：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
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

当前 v3 训练统一写入 `outputs/reward_v3_lidar40/<scene>/`，不会读取或覆盖原来位于
`outputs/<scene>/` 的 v1/v2 模型。首次训练应增加 `--fresh-start`；之后每个场景只会
在 v3 命名空间内自动加载自己的 checkpoint。三个住宅区的旋转矩形建筑会保守转换成轴对齐包围盒，
默认外扩 2 m；吴泾试飞场仍使用 8 m 外扩量。可用 `--obstacle-inflation`
覆盖当前场景的默认值。

### 安全动作、渐进目标与最佳模型

训练默认启用安全动作屏蔽：每次选动作前先预测下一步动力学，排除会立即碰撞的
动作；进入制动风险区后，还会排除继续恶化前向制动裕量的动作。随机探索和 DQN
目标动作选择均使用同一屏蔽，因此不会再把明显危险动作作为探索样本。需要做消融
实验时可增加 `--disable-safe-action-mask`。

建筑附近目标采用分场景渐进课程。吴泾仍从 30%、15--30 m 净空逐步过渡到
70%、2--12 m；三个住宅区从更容易的 10%、15--30 m 开始，在 2000 回合内逐步
过渡到“标准层”50%、5--15 m，再用 2000 回合逐步提高到“困难层”70%、2--12 m。
住宅区前期还会限制目标距离和高度，并只按建筑侧面净空采样；标准层放宽到整张地图，
困难层继续扩展目标高度（Spring Garden Phase2 从最高 40 m 渐增至 60 m）。新 checkpoint
会分别保存课程与探索进度。训练进度条中的 `near_goal` 表示最近 50 回合的实际建筑
邻近目标比例，`curriculum` 表示当前设定概率，`safe` 表示当前可选动作比例，`speed`
表示该回合平均飞行速度。

每 1000 回合默认运行 10 回合固定随机种子的纯策略验证（`epsilon=0`），按成功率、
碰撞率、成功回合平均步数、平均奖励依次比较并保存最佳模型。每 100 回合另存一次
可续训的最新 checkpoint，避免长时间 CPU 训练意外中断后损失进度。网络权重和经验回放池
保存在同一个 checkpoint 中。常规 checkpoint 保存到 v3 命名空间下的 `<scene>_dqn.pt`，
最佳模型单独保存到 `<scene>_dqn_best.pt`，不会相互覆盖。可用
`--validation-interval`、`--validation-episodes`、`--checkpoint-interval` 和
`--save-best-model` 调整；将
`--validation-interval` 设为 0 可关闭训练中验证。

默认探索率由 0.30 衰减到 0.01，衰减尺度为 500 回合。可通过
`--epsilon-start`、`--epsilon-end` 和 `--epsilon-decay` 覆盖。

```powershell
# 所有随机评估目标均优先取在建筑附近
python .\uav_drl_path_planning.py --scene wujing_airfield --skip-train `
  --eval-episodes 50 --eval-near-obstacle-probability 1.0

# 恢复为全部均匀随机目标
python .\uav_drl_path_planning.py --scene wujing_airfield --skip-train `
  --eval-episodes 50 --eval-near-obstacle-probability 0.0
```

随机评估默认使用 `--eval-goal-mode match-training`，即固定采用训练课程的“标准层”，
使测试与用于选最佳模型的验证具有相同且可重复的难度。使用 `--eval-goal-mode hard`
可测试课程的最终困难层；使用 `--eval-goal-mode stress` 可恢复旧版的无限距离/高度、
70% 建筑附近、2--12 m 三维净空压力测试。可通过
`--eval-near-obstacle-min-clearance` 和
`--eval-near-obstacle-max-clearance` 修改建筑净空范围。明确指定
`--target-x/--target-y` 时固定目标优先，不再进行随机目标采样。
课程终点可通过 `--train-near-obstacle-probability`、
`--train-near-obstacle-min-clearance`、`--train-near-obstacle-max-clearance`
修改；课程起点和时长使用对应的 `--train-near-obstacle-start-*` 和
`--train-near-obstacle-curriculum-episodes` 修改；困难层和渐进时长使用对应的
`--train-near-obstacle-hard-*` 与 `--train-near-obstacle-hardening-episodes` 修改。

```powershell
# Spring Garden Phase2：标准训练难度测试（默认）
python .\uav_drl_path_planning.py --scene spring_garden_phase2 --skip-train --eval-episodes 50

# 最终困难层 / 旧版压力测试
python .\uav_drl_path_planning.py --scene spring_garden_phase2 --skip-train `
  --eval-episodes 50 --eval-goal-mode hard
python .\uav_drl_path_planning.py --scene spring_garden_phase2 --skip-train `
  --eval-episodes 50 --eval-goal-mode stress
```

### 在已有模型基础上继续训练

不加 `--fresh-start` 时会自动加载当前场景的 `<scene>_dqn.pt`。只有同属当前 v3 奖励、
40 m 雷达配置的 checkpoint 才能安全续训。v1/v2 checkpoint 中的经验奖励无法自动重算，
而且旧模型按 24 m 雷达量程解释状态，因此升级后必须使用 `--fresh-start` 重新训练：

```powershell
python .\uav_drl_path_planning.py --scene lanxianghu_villa --episodes 5000 --eval-episodes 50 --fresh-start
```

之后由 v3 保存的 checkpoint 可以照常续训，不需要再次增加 `--fresh-start`。

默认动力学参数来自飞行日志：最大水平速度 23.0 m/s、最大三维合速度
23.18 m/s、最大上升速度 2.85 m/s、最大下降速度 1.65 m/s、最大爬升角
90°、瞬时峰值加速度 15.5 m/s²、最大减速度 3.09 m/s²。正常飞行加速度
采用给定 3--5 m/s² 区间的保守端 3.0 m/s²；Jerk 约束采用平滑后峰值
78 m/s³，并保留原始日志峰值 142 m/s³ 作为参考。26 方向雷达默认从 24 m 增加到
40 m，采样分辨率仍为 0.8 m；可用 `--lidar-range` 和 `--lidar-resolution` 覆盖。
上述参数均可通过对应命令行参数覆盖。

## v3 归一化奖励

当前密集奖励只保留四个互不重复的分项：

- `progress`：三维距离进度除以“最大速度 × 时间步”，再裁剪到 `[-1, 1]`；
- `step`：很小的固定时间成本；
- `safety_risk`：把安全半径和当前速度所需制动距离合并为一个连续风险；
- `jerk`：唯一的动作平滑正则项。

到达、碰撞和超时分别给 `+50`、`-50`、`-20`。单步进度默认最多为 `±1`，因此
碰撞惩罚不会再被按米累计的进度奖励淹没。原来的距离、绕路、转角、速度方向、
绝对/额外高度、目标高度、垂直速度、加速度、速度裁剪和悬停奖励均已移除；动力学
限幅、碰撞检测和安全动作屏蔽仍作为硬约束保留。

默认参数可以通过以下选项调整：

```text
--progress-reward-scale 1.0
--step-penalty 0.01
--safety-risk-penalty-scale 1.0
--jerk-penalty-scale 0.02
--goal-reward 50.0
--collision-penalty -50.0
--timeout-penalty -20.0
```

每一步 `info["reward_components"]` 会记录各奖励分项，
`info["reward_diagnostics"]` 会记录原始/归一化进度、净空、制动距离、安全风险和
jerk 比例。v1/v2 checkpoint 升级后必须使用 `--fresh-start` 训练新模型。

## 指定三维起点和目标点

```powershell
python .\uav_drl_path_planning.py --episodes 1500 --fresh-start --start-x 20 --start-y 100 --start-z 8 --target-x 480 --target-y 260 --target-z 12 --visualize
```

## 加载模型进行规划

```powershell
python .\uav_drl_path_planning.py --scene wujing_airfield --skip-train --load-model .\outputs\reward_v3_lidar40\wujing_airfield\models\wujing_airfield_dqn.pt --start-x 20 --start-y 100 --start-z 8 --target-x 480 --target-y 260 --target-z 12 --visualize
```

建筑水平外扩量可通过 `--obstacle-inflation` 调整，例如设为 10 m。更换场景后
应使用 `--fresh-start` 并重新训练，避免续载旧地图的模型和经验回放池。

## 吴泾离线路径数据库

`uav_offline_route_db.py` 会在吴泾建筑物外侧自动选择服务点，使用最佳 DQN 模型为
每个点生成多条候选。第 1 次使用确定性贪心策略，其余候选使用可复现的小概率安全探索；
只有成功到达、完整动力学验证通过、QGC 抽稀折线仍无碰撞的候选才参与比较。最终按
飞行时间最短、路径长度次短、奖励再次优先的规则保存一条路线。默认每户尝试 10 次、
安全探索率为 0.08，并生成 6 个住户服务点：

```powershell
python .\uav_offline_route_db.py build --resident-count 6
```

可调整每户候选数量和探索强度；固定 `--rollout-seed` 时重建结果可复现：

```powershell
python .\uav_offline_route_db.py build `
  --resident-count 6 `
  --route-attempts 20 `
  --exploration-epsilon 0.08 `
  --rollout-seed 2026
```

默认数据库为 `outputs/reward_v3_lidar40/wujing_airfield/offline_routes.sqlite`。每次构建先写入临时数据库，
全部生成结束后再原子替换正式文件，查询不会读到半成品。列出可查询住户：

```powershell
python .\uav_offline_route_db.py list
```

查询一个住户时会直接在标准输出返回 QGC 航点；增加 `--output-plan` 可同时生成能在
QGroundControl 中打开的 `.plan` 文件：

```powershell
python .\uav_offline_route_db.py query WJ-B01-E `
  --origin-lat-wgs84 31.0000000 `
  --origin-lon-wgs84 121.0000000 `
  --origin-alt-amsl 6.0 `
  --output-plan .\outputs\reward_v3_lidar40\wujing_airfield\WJ-B01-E.plan
```

示例原点必须替换为本地 ENU `(0,0,0)` 对应的现场实测 WGS-84 原点和 AMSL 海拔。
数据库同时记录每户的生成次数、成功候选数、最终入选试次和全部成功候选的时间/长度/
奖励摘要，便于追溯“为什么选中这条路线”。数据库故意不使用地图中的 GCJ-02 原点生成飞控坐标。输出的 `qgc_points` 包含任务序号、
MAVLink 命令、WGS-84 经纬度和相对高度；完整 QGC Plan 同时包含起飞、航点以及用户
显式指定的 `--end-action`。

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
  --load-model .\outputs\reward_v3_lidar40\wujing_airfield\models\wujing_airfield_dqn.pt `
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

旧版输出继续保留在 `outputs/<scene>/`。当前 v3 模型、回放池和运行结果先按实验版本、
再按场景隔离：

```text
outputs/
  wujing_airfield/                       # 旧版，保持不动
    models/wujing_airfield_dqn.pt
  reward_v3_lidar40/
    wujing_airfield/
      models/wujing_airfield_dqn.pt      # 含 v3 回放池
      models/wujing_airfield_dqn_best.pt
      offline_routes.sqlite
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
