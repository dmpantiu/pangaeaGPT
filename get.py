import os

def generate_tree(root_dir, prefix=""):
    """
    Recursively generate a tree-like list of strings representing the directory
    structure (folders and files). Folders end with a slash (/) and files are
    shown with their names.
    
    Parameters:
        root_dir (str): The path of the directory to traverse.
        prefix (str): The indentation prefix for the current level.
        
    Returns:
        list[str]: List of strings, each representing one line of the tree.
    """
    lines = []
    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        # Skip directories we don't have access to.
        return lines

    entries_count = len(entries)
    for index, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        # Use connector to mimic tree branches.
        connector = "├── " if index < entries_count - 1 else "└── "
        if os.path.isdir(path):
            lines.append(prefix + connector + entry + "/")
            extension_prefix = "│   " if index < entries_count - 1 else "    "
            # Recursively add the contents of the subfolder.
            lines.extend(generate_tree(path, prefix + extension_prefix))
        else:
            lines.append(prefix + connector + entry)
    return lines

if __name__ == "__main__":
    # Define the root directory; change "." to any directory path you need.
    root_directory = "."
    output_file = "directory_structure.txt"

    # Start with the root directory as the first line.
    tree_lines = [os.path.abspath(root_directory) + "/"]
    tree_lines.extend(generate_tree(root_directory))

    # Write the tree structure to the text file.
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))
    
    print(f"Directory structure saved to {output_file}")
