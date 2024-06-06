import os


DEFAULT_SAVE_NAME = "decoder_model"


def init_save_name(save_name=None):
    if save_name is None:
        save_name = DEFAULT_SAVE_NAME
    elif os.path.splitext(save_name)[-1] in [".c", ".cpp", ".h", ".hpp"]:
        save_name = os.path.splitext(save_name)[0]
    return os.path.basename(save_name)


def init_save_dir(save_dir, sub_dir):
    save_dir = os.path.abspath(save_dir)
    if os.path.basename(save_dir) in ["model", "layer"]:
        save_dir = os.path.dirname(save_dir)
    save_dir = os.path.join(save_dir, sub_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir