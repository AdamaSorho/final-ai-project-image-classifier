import argparse

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_directory', nargs=1)
    parser.add_argument('checkpoint', nargs=1)
    parser.add_argument('--category_names', default="cat_to_name.json")
    parser.add_argument('--top_k', default=3, type=int)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    
    return args
