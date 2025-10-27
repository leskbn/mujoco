from dm_control import suite

env = suite.load("cartpole", "swingup")  # MuJoCo 물리
ts = env.reset()
for _ in range(1000):
    action = env.action_spec().generate_value()
    ts = env.step(action)
