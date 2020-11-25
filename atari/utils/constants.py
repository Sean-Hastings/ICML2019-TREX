MUJOCO_ENVS = ['ant', 'hopper', 'halfcheetah', 'humanoid', 'pusher', 'reacher', 'striker', 'swimmer', 'thrower', 'walker']

CHECKPOINT_DICT = {
    'enduro': (3100, 3650, 4450, 50),
    'seaquest': (5, 35, 70, 5),
    'hero': (300, 1500, 2400, 50),
    'mujoco': (40, 320, 480, 20),
    'other': (100, 1050, 1450, 50)
}

def get_env_id_type(env_name):
    env_type = "atari"

    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    elif env_name == "halfcheetah":
        env_id = "HalfCheetah-v2"
        env_type = 'mujoco'
    elif env_name in MUJOCO_ENVS:
        env_id = env_name[0].upper() + env_name[1:] + "-v2"
        env_type = 'mujoco'
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    return env_id, env_type


def get_checkpoint_range(env_name, demo=True):
    if demo:
        _min, _max, _step = get_checkpoints_demos(env_name)
    else:
        _min, _max, _step = get_checkpoints_extrapolate(env_name)

    return range(_min, _max + _step, _step)


def get_checkpoints_demos(env_name):
    _min, _max, _, _step = CHECKPOINT_DICT['other']
    for key in CHECKPOINT_DICT.keys():
        if env_name in key:
            _min, _max, _, _step = CHECKPOINT_DICT[key]
            break

    return _min, _max, _step


def get_checkpoints_extrapolate(env_name):
    _, _min, _max, _step = CHECKPOINT_DICT['other']
    for key in CHECKPOINT_DICT.keys():
        if env_name in key:
            _, _min, _max, _step = CHECKPOINT_DICT[key]
            break

    return _min, _max, _step
