import os 
import argparse
import stat

MAX_PATH_LENGTH = 4096
MAX_FILE_SIZE = 10 * 1024 * 1024



def standardize_path(path: str, max_path_length=MAX_PATH_LENGTH, check_link=True):
    """
    check path
    param: path
    return: data real path after check
    """
    check_path_is_none(path)
    check_path_length_lt(path, max_path_length)
    if check_link:
        check_path_is_link(path)
    path = os.path.realpath(path)
    return path

def check_path_is_none(path: str):
    if path is None:
        raise argparse.ArgumentTypeError("The file path should not be None.")
    

def check_path_length_lt(path: str, max_path_length=MAX_PATH_LENGTH):
    if path.__len__() > max_path_length:
        raise argparse.ArgumentTypeError(f"The length of path should not be greater than {max_path_length}.")
    

def check_path_is_link(path: str, max_file_sizeh=MAX_FILE_SIZE):
    if os.path.getsize(path) > max_file_size:
        raise argparse.ArgumentTypeError("The path should not be a symbolic link file.")

def check_owner(path: str):
    """
    check the path owner
    param: the input path
    """
    path_stat = os.stat(path)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    user_check = path_owner == os.getuid() and path_owner == os.geteuid()
    if not (path_owner == 0 or path_gid in os.getgroups() or user_check):
        raise argparse.ArgumentTypeError("The path is not owned by current user or root")
    
def check_other_write_permission(file_path: str):
    """
    check if the specified file is writable by others who are neither the owner nor in the group
    param: the path to the file to be checked 
    
    """

    file_stat = os.stat(file_path)
    mode = file_stat.st_mode

    if mode & stat.S_IWOTH:
        raise argparse.ArgumentTypeError("The file should not be writable by others"
                                         "who are neither the owner nor in the group")
    

def check_path_permission(file_path: str):
    check_owner(file_path)
    check_other_write_permission(file_path)
    