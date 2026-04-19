# 第一题：热连轧带钢浪形检测软件（可运行实现）

本项目按 `需求分析文档.pdf` 的第一题要求实现了一个可运行版本：
- 视频输入与实时帧处理
- 去雾/增强/滤波预处理
- 带钢主轮廓提取
- 厚度、宽度估算（可标定到 mm）
- 浪形识别：`DS单边浪 / WS单边浪 / 双边浪 / 中间浪` + `0~3` 等级
- 导出标注视频、逐帧CSV、NG图片与运行报告

## 目录结构

```text
q1_wave_detect/
  configs/default.yaml
  scripts/run_detector.py
  src/wave_detector/
  results/
```

## 环境准备

```bash
cd /home/tlx/IndustrialSoftware/q1_wave_detect
python -m pip install -r requirements.txt
```

## 运行示例

```bash
cd /home/tlx/IndustrialSoftware/q1_wave_detect
PYTHONPATH=src python scripts/run_detector.py \
  --video "/home/tlx/IndustrialSoftware/样例视频及识别需求/机架间浪形识别视频/6-7/F6_第一块还有双变浪，且有DS的单边浪，第二块有双边浪.mp4" \
  --config configs/default.yaml \
  --output results/f6_demo
```

## 输出内容

- `annotated.mp4`：叠加检测框、尺寸与浪形等级
- `results.csv`：逐帧结果（包含延迟、四类浪形等级、mm估计值）
- `results.csv` 中 `strip_present` 表示当前帧是否检测到带钢主体（用于过滤空载帧）
- `ng_frames/*.jpg`：发生浪形告警的截图
- `summary.json`：统计信息（平均/95分位/最大耗时、NG帧数）

## 参数说明（关键）

- `mm_per_px_x/mm_per_px_y`：像素转毫米比例，需现场标定后替换
- `thr_percentile`：分割阈值强度
- `rms_mm_thresholds`：`0/1/2/3` 等级阈值
- `roi_*`：处理区域，建议按机架相机实际安装位置微调

## 与需求文档的对应关系

- 图像采集：支持本地视频流输入（可扩展为 RTSP）
- 图像预处理：CLAHE 增强 + 高斯滤波
- 边界检测：阈值分割 + 轮廓提取
- 浪形识别：基于边缘轮廓平直度 RMS 进行 DS/WS/双边/中间浪判别
- 标注与导出：实现
- 性能统计：输出单帧处理耗时，便于验证 `<200ms` 指标

## 说明

当前实现为课程大作业的工程化基线版本，重点是完成“端到端可运行 + 指标可评估 + 可持续迭代”。
若你后续要冲更高精度，可在该框架上替换为深度学习分割或时序模型。

## 已生成示例结果

- `results/no_wave_demo_v3`：无浪形视频小样，`ng_frames=0`
- `results/f6_full_v3`：全量视频检测结果，包含标注区间附近告警
- `results/f6_vis/annotated.mp4`：可视化演示视频（前 1400 帧）
