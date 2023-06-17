import os
import subprocess

def split(path, size):
    for root, dirs, files in os.walk(path):
        if root.endswith("bass"):
            if "archive_bass.wav" in files or "archive_residuals.wav" in files:
                # which means the file has been splited
                pass
            else:
                for file in files:
                    name = os.path.join(root, file)
                    CMD_split = ['sox', name, name, 'trim', '0', str(size), ':', 'newfile', ':', 'restart']
                    subprocess.run(CMD_split)
                    # rename the archive file
                    os.rename(name, os.path.join(root, "archive_" + file))

def reverse_split(path):
    for root, dirs, files in os.walk(path):
        if root.endswith("bass"):
            if "archive_bass.wav" in files or "archive_residuals.wav" in files:
                for file in files:
                    if not file.startswith("archive"):
                        os.remove(os.path.join(root, file))
                        # print(os.path.join(root, file))
                #  rename the archive file back
                for file in files:
                    if file.startswith("archive"):
                        os.rename(os.path.join(root, file), os.path.join(root, file[8:]))
                        # print(os.path.join(root, file))

# reverse_split(path)
# split(path, 30)