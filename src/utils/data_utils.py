import os

# def num_of_classes(directory):
#     return len(next(os.walk(directory))[1])

def num_of_classes(directory):
    if directory is None:
        return 1000  
    try:
        return len([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    except Exception as e:
        print(f"Error determining number of classes: {e}")
        return 1000  # def 100

def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))
                print(f"Removed: {os.path.join(root, file)}")
