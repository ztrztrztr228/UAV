# 三个独立住宅区估算地图

本目录保存三个住宅区的原始旋转矩形场景定义。训练时由
`uav_drl/scenes.py` 转换成保守的轴对齐三维包围盒，再接入统一的
`uav_drl_path_planning.py --scene ...` 训练入口。

## 数据性质

三个场景根据高德公开网络地图、卫星图、公开楼盘地址信息及项目 PDF 中的尺寸
说明联合近似。坐标、朝向和轮廓不是测绘成果，所有场景均标记为
`network_map_estimate_not_for_real_flight`，只能用于算法仿真。

- `lanxianghu_villa_map.py`：兰香湖贰号东/西区弧形别墅排布，115 个估算屋顶障碍物，高 10 m。
- `sanming_garden_map.py`：三明花园规则组合楼栋，30 个组合障碍物，高 20 m。
- `spring_garden_phase2_map.py`：春天花园二期连排高层，20 个长条组合障碍物，高 60 m。

所有建筑均采用带 `yaw_deg` 的旋转矩形，在 `geometry.py` 中可转换为局部坐标
GeoJSON。地图原点采用 GCJ-02 估算值，但导出的建筑坐标是局部 ENU 米制坐标，
不能直接与 WGS-84 GPS 坐标混用。

## 生成预览和数据

在仓库根目录运行：

```powershell
python -m standalone_maps.preview_all
```

结果写入 `outputs/standalone_maps/`，每个场景包含参数 JSON、局部坐标 GeoJSON
以及二维/三维预览 PNG。

## 运行独立测试

```powershell
python -m unittest standalone_maps.test_maps -v
```

训练命令及按场景隔离的模型、运行结果目录请参阅仓库根目录 `README.md`。
