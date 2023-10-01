import json

def get_names(category_names_dir):
    with open(category_names_dir, 'r') as f:
        names = json.load(f)
    
    return names
