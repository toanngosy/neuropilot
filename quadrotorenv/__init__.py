from gym.envs.registration import register

register(
    id='QuadRotorEnv-v0',
    entry_point='quadrotorenv.quadrotorenv:QuadRotorEnv',
)
