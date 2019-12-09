import glob
import os

def get_latest_run_id(save_path, dir_name):
    """
    returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(save_path + "/{}_[0-9]*".format(dir_name)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if dir_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id

def get_log_dir(env, algo):
    log_dir = os.getcwd() + '/data/' + env + '/'
    latest_run_id = get_latest_run_id(log_dir, algo)
    log_dir = os.path.join(log_dir, "{}_{}".format(algo, latest_run_id+1))
    if os.path.exists(log_dir):
        raise ValueError('Log directory is already exists')
    return log_dir

def run_background(f, stop):
    while not stop.is_set():
        f()
