# 《基于相关滤波与孪生网络的无人机视觉目标跟踪方法研究》-ReSiam跟踪方法

此文件夹为基于异常状态感知的无人机目标跟踪方法（ReSiam）源代码。

运行环境为：

- python 3.9
- pytorch 1.10
- CUDA 11.6

运行方式：

1. 调用摄像头

   ```bash
   python ./tools/run_webcam.py --tracker_name resiam --tracker_param resiam --debug 0
   ```

2. 跟踪单一图像序列

   ```bash
   python ./tools/run_tracker.py --tracker_name resiam --tracker_param resiam --dataset_name uav123 --sequence bike1 --debug 1 --threads 0
   ```

3. 对数据集中所有序列进行跟踪

   ```bash
   python ./tools/run_experiment.py --experiment_module resiam --experiment_name resiam_uav123 --debug 0 --threads 0
   ```
