from pathlib import Path

def get_project_root():
    """
    Find the root directory of the project by searching for 'pyproject.toml'.

    Returns:
        Path: The absolute path to the project root directory containing 'pyproject.toml'.

    Notes:
        - This assumes that 'pyproject.toml' exists in the root of the project.
        - Traverses parent directories until the file is found.
    """
    p = Path(__file__).resolve()
    while not (p / "pyproject.toml").exists():
        p = p.parent
    return p

def get_data_path():
    """
    Get the absolute path to the 'data' directory located at the project root.

    Returns:
        Path: The absolute path to the 'data' directory.

    Notes:
        - This relies on 'get_project_root()' to determine the base path.
        - Does not create the directory if it doesn't exist; caller is responsible for that.
    """
    root = get_project_root()
    return root / 'data'
