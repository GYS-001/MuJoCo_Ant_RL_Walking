"""
Phase 3 v2: 温和调参 —— 只调整 reward 鼓励速度 + 轻量能耗惩罚
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np

class GentleAntReward(gym.RewardWrapper):
    """轻量修改：保持默认 reward 结构，只做微调"""
    def reward(self, reward):
        # 默认 reward 已经包含：前进速度 + 存活奖励 - 控制代价 - 接触代价
        # 我们只加一点点额外的东西

        # ① 身体高度温和引导：鼓励站直，不强制
        height = self.unwrapped.data.qpos[2]
        height_bonus = max(0, height - 0.3) * 0.5  # 高于 0.3m 就加分

        # ② 鼓励步态对称（左右腿用力差不多）
        ctrl = self.unwrapped.data.ctrl
        left_sum = abs(ctrl[[0,1,2,4,5,6]]).sum()   # 左边 6 个关节
        right_sum = abs(ctrl[[3,7]]).sum()           # 右边 2 个
        # 不强制，只做轻量引导
        symmetry_bonus = -abs(left_sum/6 - right_sum/2) * 0.02

        return reward + height_bonus + symmetry_bonus

def train():
    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = GentleAntReward(env)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,     # 介于 Phase2(3e-4) 和失败版(1e-4) 之间
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,        # 介于 0.2 和 0.1 之间
        ent_coef=0.001,         # 极小探索（Phase2 是 0，失败版是 0.01）
        verbose=1,
        tensorboard_log="./tb_logs/tuned_v2/",
        device="cuda"
    )

    print("\n" + "=" * 50)
    print("Phase 3 v2: 温和调参")
    print("期望：ep_rew_mean > Phase2(524)，步态更自然")
    print("=" * 50 + "\n")

    model.learn(total_timesteps=300_000)

    model.save("ant_walk_tuned_v2")
    print("\n模型已保存: ant_walk_tuned_v2.zip")

    # 测试
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
    print(f"\n训练后测试奖励: {total:.1f}")
    print(f"Phase 2 参考: ep_rew_mean=524, test=124")

if __name__ == "__main__":
    train()