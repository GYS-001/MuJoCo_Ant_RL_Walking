"""
Phase 3: 调 Reward Function + PPO 超参数，让 Ant 走得更快更稳
"""
import gymnasium as gym
from gymnasium import RewardWrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ========== 自定义 Reward ==========
class TunedAntReward(RewardWrapper):
    """在默认 reward 基础上加三个惩罚项"""
    def reward(self, reward):
        # ① 能耗惩罚：电机用力的平方和，鼓励节能步态
        ctrl_cost = np.sum(np.square(self.unwrapped.data.ctrl))
        # ② 姿态惩罚：身体高度偏离目标（防止趴着蹭）
        height = self.unwrapped.data.qpos[2]  # z 坐标
        height_penalty = max(0, 0.75 - height) * 5  # 低于 0.75m 扣分
        # ③ 横向漂移惩罚：侧向速度不应该太大
        vel_y = self.unwrapped.data.qvel[1]
        lateral_penalty = abs(vel_y) * 2

        return reward - 0.1 * ctrl_cost - height_penalty - lateral_penalty

def train_tuned():
    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = TunedAntReward(env)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,      # 调低：更稳定
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,          # 调小：更新更保守
        ent_coef=0.01,           # 加点探索（原来是 0）
        verbose=1,
        tensorboard_log="./tb_logs/tuned/",
        device="cuda"
    )

    print("\n" + "=" * 50)
    print("Phase 3 训练：自定义 Reward + 调参")
    print("期望：ep_rew_mean > 800，步态更自然")
    print("=" * 50 + "\n")

    model.learn(total_timesteps=300_000)

    model.save("ant_walk_tuned")
    print("\n模型已保存: ant_walk_tuned.zip")

    # 快速测试
    env_test = gym.make("Ant-v5", render_mode="rgb_array")
    obs, _ = env_test.reset()
    total = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env_test.step(action)
        total += reward
        if term or trunc:
            break
    env_test.close()
    print(f"调参后测试奖励: {total:.1f} (Phase 2: 124)")

if __name__ == "__main__":
    train_tuned()