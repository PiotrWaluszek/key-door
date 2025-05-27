from gymnasium.envs.registration import register

register(
    id="gymnasium_env/KeyDoorEnv",
    entry_point="gymnasium_env.envs:KeyDoorEnv",
)
