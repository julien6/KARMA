from gym.envs.registration import register

register(
    id='Redis-v0',
    entry_point='k8s_hpa.envs:Redis',
)