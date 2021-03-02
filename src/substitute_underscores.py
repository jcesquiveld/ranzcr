import os

def rename_underscores(dir):
    files = os.listdir(dir)
    for file in files:
        os.rename(os.path.join(dir, file), os.path.join(dir, file.replace('=', '_')))

if __name__ == '__main__':

    dir = '../models/coworker_2'
    rename_underscores(dir)