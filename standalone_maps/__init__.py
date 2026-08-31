"""与现有 UAV 训练代码隔离的住宅区场景估算模块。"""

# 对外只暴露通用建筑和场景数据结构；具体地图由各自模块提供。
from .geometry import Building, SceneMap

__all__ = ["Building", "SceneMap"]
