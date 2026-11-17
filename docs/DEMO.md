# 看图与复现

## 仓库内的预览

- [基础规划对比](assets/basic.png)：采样规划与弹性带。
- [动态交叉场景](assets/crossing.png)：基础 MPPI 与风险代价 MPPI。
- [走廊场景](assets/corridor.png)：移动障碍下的闭环重规划。
- [重规划过程图](assets/crossing-process.svg)：候选轨迹、选中轨迹与执行历史。
- [动态交叉动画](assets/crossing-animation.svg)：下载后用支持 SVG 动画的浏览器打开。

GitHub 可能只显示 SVG 动画的静态帧。下载文件在浏览器中打开，或运行演示并打开本地 `animation.svg`。

## 生成同类展示

```bash
python3 -B -m planner_lab.run --out outputs/basic
python3 -B -m planner_lab.run_mppi \
  --out outputs/mppi --scenarios crossing,corridor \
  --seeds 1 --max-steps 130 --rollouts 44 --horizon 16
```

不依赖额外图片工具即可得到 SVG。若希望导出 README 中同类 PNG，可选用系统工具 `rsvg-convert`：

```bash
rsvg-convert outputs/mppi/crossing/comparison.svg -o outputs/crossing.png
```

`docs/demo/` 中仅保留本次展示运行的轻量配置与指标；原始轨迹与逐步候选可由命令重新生成。固定随机种子便于比较轨迹，计算耗时则会因机器负载而变化。
