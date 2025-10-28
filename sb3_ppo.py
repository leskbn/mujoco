# SB3 PPO × dm_control(MuJoCo physics) — 태스크별 프리셋 + 학습/평가/뷰어
import numpy as np
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# ✅ shimmy 래퍼 (dm_control → Gymnasium)
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

# --------- 여기만 바꾸면 됨 ----------
DOMAIN, TASK = (
    "ball_in_cup",
    "catch",
)  # 예) ("finger","spin"), ("cheetah","run"), ("walker","walk"), ("ball_in_cup","catch")
SEED = 0
# -------------------------------------

# 태스크별 권장 프리셋
PRESETS = {
    ("cartpole", "balance"): dict(
        total=100_000,
        n_steps=1024,
        batch=64,
        lr=3e-4,
        ent=0.0,
        clip=0.2,
        net=(128, 128),
    ),
    ("ball_in_cup", "catch"): dict(
        total=1_000_000,
        n_steps=1024,
        batch=64,
        lr=3e-4,
        ent=0.0,
        clip=0.2,
        net=(256, 256),
    ),
    ("finger", "spin"): dict(
        total=500_000,
        n_steps=2048,
        batch=128,
        lr=3e-4,
        ent=0.01,
        clip=0.2,
        net=(256, 256),
    ),
    ("cheetah", "run"): dict(
        total=2_000_000,
        n_steps=4096,
        batch=256,
        lr=1e-4,
        ent=0.0,
        clip=0.2,
        net=(256, 256),
    ),
    ("walker", "walk"): dict(
        total=2_000_000,
        n_steps=4096,
        batch=256,
        lr=3e-4,
        ent=0.0,
        clip=0.2,
        net=(256, 256),
    ),
    ("hopper", "hop"): dict(
        total=2_000_000,
        n_steps=4096,
        batch=256,
        lr=3e-4,
        ent=0.0,
        clip=0.2,
        net=(256, 256),
    ),
    ("cartpole", "swingup"): dict(
        total=1_000_000,
        n_steps=2048,
        batch=128,
        lr=3e-4,
        ent=0.01,
        clip=0.2,
        net=(256, 256),
    ),
}
cfg = PRESETS[(DOMAIN, TASK)]
np.random.seed(SEED)


def make_env():
    # dm_control 원본 → shimmy로 Gymnasium 호환 → 관측 평탄화
    dmc = suite.load(DOMAIN, TASK, task_kwargs={"random": SEED})
    env = DmControlCompatibilityV0(dmc)
    env = FlattenObservation(env)
    return env


# VecEnv + 관측 정규화
venv = DummyVecEnv([make_env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

policy_kwargs = dict(net_arch=[dict(pi=list(cfg["net"]), vf=list(cfg["net"]))])
model = PPO(
    "MlpPolicy",
    venv,
    n_steps=cfg["n_steps"],
    batch_size=cfg["batch"],
    learning_rate=cfg["lr"],
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=cfg["clip"],
    ent_coef=cfg["ent"],
    vf_coef=0.5,
    seed=SEED,
    verbose=1,
    policy_kwargs=policy_kwargs,
)

print(f"[TRAIN] {DOMAIN}/{TASK} with {cfg}")
model.learn(total_timesteps=cfg["total"])

# ===== 평가(1 에피소드) =====
venv.training = False
try:
    venv.seed(SEED + 1)  # 버전에 따라 없을 수 있음
except Exception:
    pass

obs = venv.reset()  # ⬅️ seed 인자 없이
ep_ret, ep_len = 0.0, 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    # ⬇⬇⬇ VecEnv는 4개 반환: (obs, rewards, dones, infos)
    obs, rewards, dones, infos = venv.step(action)
    ep_ret += float(rewards[0])  # 벡터형 반환이라 [0]로 꺼냄
    ep_len += 1
    if bool(dones[0]):
        break
print(f"[EVAL] return={ep_ret:.1f}, len={ep_len}")

# ===== (옵션) dm_control 뷰어 10초 데모 =====
#  - 맥: export MUJOCO_GL=glfw
from dm_control import viewer as dm_viewer


def obs2flat(ts):
    return np.concatenate([v.ravel() for v in ts.observation.values()]).astype(
        np.float32
    )


def normalize_obs(flat_obs):
    rms = venv.obs_rms
    if rms is None or rms.mean is None or rms.var is None:
        return flat_obs
    return (flat_obs - rms.mean) / np.sqrt(rms.var + 1e-8)


def demo_policy(ts):
    flat = obs2flat(ts)
    flat_n = normalize_obs(flat)
    act, _ = model.predict(flat_n, deterministic=True)
    return act.astype(np.float32)


print("[DEMO] viewer 창(10초). Space=재생/일시정지, H=도움말")
dm_viewer.launch(
    suite.load(DOMAIN, TASK, task_kwargs={"random": SEED + 2}), policy=demo_policy
)
