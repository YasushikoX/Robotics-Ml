import tensorflow as tf
import numpy as np
import os

# Parameters
data_dir = r'C:\Users\yasus\Projects\Robotics-Ml\TrainData\npy_files' # directory where the data is stored
num_classes = 2 # number of classes in the data
img_rows, img_cols, img_depth = 144, 256, 3 # input shape for C3D

def load_and_preprocess_data(file_path):
    # load the data
    data = np.load(file_path)
    print(f"Loaded data shape: {data.shape}")
    
    # preprocess the data
    data = (data - np.mean(data)) / np.std(data)
    data = np.reshape(data, (img_rows, img_cols, img_depth, 1))
    print(f"Preprocessed data shape: {data.shape}")
    
    return data

def load_data_and_labels(data_dir, num_classes):
    # load the filenames
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.npy')]
    print(f"Found {len(filenames)} files in {data_dir}")
    
    # load the data and labels
    data = [load_and_preprocess_data(f) for f in filenames]
    labels = [int(os.path.basename(f).split('_')[0]) for f in filenames]
    labels = tf.keras.utils.to_categorical(labels, num_classes)
    print(f"Labels shape: {labels.shape}")
    
    return np.array(data), np.array(labels)

# load the data and labels
x_train, y_train = load_data_and_labels(data_dir, num_classes)

# save the preprocessed data
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)

print("Preprocessing done!")