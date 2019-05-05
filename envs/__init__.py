from gym.envs.registration import register

register(
    id='QuadRotorEnv-v0',
    entry_point='envs.quadrotorenv:QuadRotorEnv_v0'
)

register(
    id='QuadRotorEnv-v1',
    entry_point='envs.quadrotorenv:QuadRotorEnv_v1'
)

register(
    id='SimpleTargetEnv2D-v0',
    entry_point='envs.simpletargetenv:SimpleTargetEnv2D'
)

register(
    id='SimpleTargetEnv3D-v0',
    entry_point='envs.simpletargetenv:SimpleTargetEnv3D'
)
