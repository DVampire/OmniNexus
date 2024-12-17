import os


def get_project_root():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.dirname(path)  # get to parent, outside of project code path"
    return path


def assemble_project_path(path):
    """Assemble a path relative to the project root directory"""
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path
