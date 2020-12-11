import os, sys

root_folder = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

def relpath(*dirs):
    return os.path.relpath(os.path.join(root_folder, *dirs), os.getcwd())

sys.path.insert(0, os.path.join(root_folder, 'scripts'))
