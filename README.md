# 🦿 MuJoCo Ant 四足机器人强化学习行走

> 使用 **PPO（Proximal Policy Optimization）** 在 MuJoCo 物理仿真中训练四足机器人 Ant
> 从零学会行走。重点验证了 **奖励函数设计**对策略行为的关键影响。

<p align="center">
  <img src="ant_compare.png" width="80%">
</p>

## 概述

| 实验 | 奖励函数 | 测试 Reward | 行为 |
|------|---------|:---:|------|
| Phase 2 | MuJoCo 默认（前进速度 + 存活 - 能耗） | 124 | 能走但蹭地 |
| Phase 3 v1 | 加高度强制 + 横向漂移惩罚 | 2.3（翻车） | 策略瘫痪 |
| **Phase 3 v2** | **温和站姿引导 + 对称性鼓励** | **391** | **步态高、协调** |

## 核心发现

1. **Reward Function 是 RL 的灵魂**：过强的约束（高度强制、方向惩罚）导致
   策略不敢探索，训练完全失败；温和的 bonus 引导保留了默认 reward 的有效信号，
   同时推动策略向期望行为收敛。

2. **PPO 超参数的敏感度**：`ent_coef`（探索系数）从 0 → 0.01 带来了策略过度
   随机化（std 从 0.2 飙升到 0.98）；`clip_range` 和 `learning_rate` 需要协同调节。

3. **训练 reward ≠ 真实表现**：Phase 2 训练 reward 更高（524 vs 38），但测试时
   Phase 3 v2 高出 3 倍（391 vs 124）——说明默认 reward 被策略投机利用
   （学会了蹭地拿分），调参后的 reward 引导出了真正的协调行走。

## 技术栈

- **RL 算法**: PPO（Stable-Baselines3）
- **物理引擎**: MuJoCo 3.10
- **环境**: Gymnasium Ant-v5（8 关节四足机器人）
- **硬件**: NVIDIA GeForce RTX 4060 Laptop GPU

## 环境

```bash
conda create -n rl_walk python=3.10 -y
conda activate rl_walk
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "stable-baselines3[extra]" mujoco gymnasium moviepy imageio
```
## 运行
```bash
python train_ant.py           # Phase 2: 默认训练
python train_ant_tuned_v2.py  # Phase 3: 调 reward 训练
python record_v2.py           # 录制演示视频
```
| 随机策略 | Phase 2 默认训练 | Phase 3 v2 调参后 |
| :---: | :---: | :---: |
| [▶] ant_random.mp4 | [▶] ant_trained.mp4 | [▶] ant_tuned_v2.mp4 |
