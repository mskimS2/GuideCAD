import os
import subprocess
from tqdm import tqdm


def extract_folder_and_files(directory):
    folder_list = []
    
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            file_list = os.listdir(folder_path)
            folder_list.append({"folder": folder, "files": file_list})

    return folder_list


def create_directory_if_not_exists(image_root):
    if not os.path.exists(image_root):
        os.makedirs(image_root)


def run_parallel_subprocess(commands):
    processes = []
    for cmd in commands:
        p = subprocess.Popen(cmd)
        processes.append(p)

    for p in processes:
        p.wait()


if __name__ == "__main__":
    root = "dataset/data/cad_vec"
    folder_data = extract_folder_and_files(root)

    folder_data = folder_data[50:]
    batch_size = 4
    for data in tqdm(folder_data, ncols=100, total=len(folder_data)):
        batch_count = 0
        commands = []
        folder_name = data["folder"]
        for file_name in data["files"]:
            h5file = f"dataset/data/cad_vec/{folder_name}/{file_name}"
            image_root = f"dataset/data/cad_image/{folder_name}/{file_name.split('.')[0]}"
            create_directory_if_not_exists(image_root)

            cmd = ["python", "worker_script.py", h5file, image_root]
            commands.append(cmd)
            batch_count += 1

            if batch_count == batch_size:
                run_parallel_subprocess(commands)
                commands = []
                batch_count = 0

        if len(commands) > 0:
            run_parallel_subprocess(commands)
