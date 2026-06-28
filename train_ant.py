"""
Phase 2: 用 SB3 PPO 训练 MuJoCo Ant-v5 四足机器人学会走路
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

def train():
    # ========== 创建环境 ==========
    # Ant-v5: 四足蚂蚁，8个关节，目标是用四条腿往前走
    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = Monitor(env)  # 记录 reward 曲线

    # ========== 创建 PPO 模型 ==========
    model = PPO(
        "MlpPolicy",           # 策略网络：多层感知机
        env,
        learning_rate=3e-4,    # 学习率
        n_steps=2048,          # 每轮收集多少步数据
        batch_size=64,         # 每次更新用的 batch 大小
        n_epochs=10,           # 每批数据学几遍
        gamma=0.99,            # 折扣因子（远期奖励的重要性）
        gae_lambda=0.95,       # GAE 平滑参数
        clip_range=0.2,        # PPO clip 范围
        ent_coef=0.0,          # 探索系数
        verbose=1,             # 打印训练日志
        tensorboard_log="./tb_logs/",
        device="cuda"          # 用你的 RTX 4060
    )

    # ========== 训练 ==========
    print("\n" + "=" * 50)
    print("开始训练 Ant-v5，目标：四肢协调前进")
    print("观察 reward 曲线：上升 = 学会走路，不变 = 还在抽搐")
    print("=" * 50 + "\n")

    model.learn(total_timesteps=500_000)  # 50万步，约20-30分钟

    # ========== 保存模型 ==========
    model.save("ant_walk_ppo")
    print("\n模型已保存: ant_walk_ppo.zip")

    # ========== 演示（录视频）==========
    print("\n录制演示视频...")
    from gymnasium.wrappers import RecordVideo
    env_video = gym.make("Ant-v5", render_mode="rgb_array")
    env_video = RecordVideo(
        env_video,
        video_folder="./videos",
        episode_trigger=lambda x: x == 0,  # 只录第一回合
        name_prefix="ant_trained"
    )

    obs, _ = env_video.reset()
    total_reward = 0
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_video.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env_video.close()
    print(f"视频录制完成: ./videos/ant_trained*.mp4")
    print(f"演示回合总奖励: {total_reward:.1f}")

    # ========== 对比：随机策略 ==========
    print("\n录制随机策略对比视频...")
    env_random = gym.make("Ant-v5", render_mode="rgb_array")
    env_random = RecordVideo(
        env_random,
        video_folder="./videos",
        episode_trigger=lambda x: x == 0,
        name_prefix="ant_random"
    )

    obs, _ = env_random.reset()
    total_random = 0
    for step in range(1000):
        action = env_random.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env_random.step(action)
        total_random += reward
        if terminated or truncated:
            break

    env_random.close()
    print(f"随机策略总奖励: {total_random:.1f} (vs 训练后: {total_reward:.1f})")
    print("\n训练完成！对比两个视频看看区别")

if __name__ == "__main__":
    train()