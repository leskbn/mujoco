# SB3 PPO on dm_control CartPole via shimmy (no Gymnasium envs)
import numpy as np
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

# ✅ 올바른 임포트 경로
from shimmy.dm_control_compatibility import DmControlCompatibilityV0  # <-- 여기!

# 문서: shimmy.dm_control_compatibility.DmControlCompatibilityV0  [oai_citation:1‡Shimmy Documentation](https://shimmy.farama.org/environments/dm_control/?utm_source=chatgpt.com)

TASK = "balance"  # "swingup"으로 바꿔도 됨(학습 더 오래 걸림)
SEED = 0

# 1) 원본 dm_control env → shimmy로 Gymnasium API 변환 → 관측 flatten
dmc_env = suite.load("cartpole", TASK, task_kwargs={"random": SEED})
env = DmControlCompatibilityV0(dmc_env)  # Gymnasium Env로 래핑
env = FlattenObservation(env)  # dict obs → 1D 벡터

# 2) 학습 (빠른 테스트용 하이퍼)
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    seed=SEED,
    verbose=1,
)
model.learn(total_timesteps=50_000)  # 50k~200k에서 시작해보고 조절

# 3) 간단 평가(1 에피소드)
obs, info = env.reset(seed=SEED + 1)
ep_ret, ep_len = 0.0, 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    ep_ret += float(reward)
    ep_len += 1
    if terminated or truncated:
        break
print(f"[EVAL] {TASK}: return={ep_ret:.1f}, len={ep_len}")

# 4) (옵션) dm_control 뷰어로 10초 데모
#    맥에선 한 번:  export MUJOCO_GL=glfw
from dm_control import viewer as dm_viewer


# dm_control viewer는 원래 환경(ts)->action 콜백을 받음
def obs2vec(ts):
    # dm_control의 dict 관측을 학습 때와 동일하게 평탄화
    return np.concatenate([v.ravel() for v in ts.observation.values()]).astype(
        np.float32
    )


def demo_policy(ts):
    flat_obs = obs2vec(ts)  # FlattenObservation과 동일 형태
    action, _ = model.predict(flat_obs, deterministic=True)
    return action.astype(np.float32)  # dm_control 액션 스페이스에 맞게 전달


print("[DEMO] viewer 창이 뜹니다. (스페이스: 재생/일시정지, H: 도움말)")
dm_viewer.launch(
    suite.load("cartpole", TASK, task_kwargs={"random": SEED + 2}), policy=demo_policy
)
