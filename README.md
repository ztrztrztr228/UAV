# UAV 三维深度强化学习轨迹规划

本项目使用 DQN 实现无人机在三维小区地图中的自主轨迹规划。输入三维起点和目标点后，智能体会在长方体建筑物和地图边界约束下学习飞向目标点。

## 项目结构

```text
uav_drl_path_planning.py       # 主函数和命令行的入口
uav_drl/
  actions.py                   # 三维动作空间，26 个方向 + hover
  config.py                    # 三维地图、建筑物、环境参数
  environment.py               # 三维环境、状态空间、奖励函数、碰撞检测
  model.py                     # DQN 神经网络
  agent.py                     # DQN 智能体和经验回放池
  training.py                  # 训练和评估流程
  visualization.py             # 训练曲线、三维轨迹图、CSV 保存
  utils.py                     # 随机种子、三维坐标解析
UAV_DRL_项目提纲.md             # 项目说明和报告提纲
```

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
python .\uav_drl_path_planning.py --episodes 1500 --eval-episodes 20 --visualize
```

## 指定三维起点和目标点

```powershell
python .\uav_drl_path_planning.py --episodes 1500 --start-x 5 --start-y 5 --start-z 8 --target-x 92 --target-y 88 --target-z 12 --visualize
```

## 加载模型进行规划

```powershell
python .\uav_drl_path_planning.py --skip-train --load-model .\outputs\uav_dqn.pt --start-x 5 --start-y 5 --start-z 8 --target-x 92 --target-y 88 --target-z 12 --visualize
```

## 输出文件

训练和评估结果默认保存在 `outputs/` 下，包括模型文件、训练曲线、最佳轨迹 CSV 和三维轨迹图。该目录已加入 `.gitignore`，默认不会提交到仓库。
