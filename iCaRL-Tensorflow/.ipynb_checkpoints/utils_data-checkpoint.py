import tensorflow as tf
import numpy as np
import os
import scipy.io
import sys
import glob
try:
    import cPickle
except:
    import _pickle as cPickle

def parse_devkit_meta(devkit_path):
    meta_mat                = scipy.io.loadmat(devkit_path+'/meta.mat')
    #print("meta_mat['synsets']:", meta_mat['synsets'])
    labels_dic              = dict((m[0][1][0], m[0][0][0][0]-1) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
   # print("Labels dictionary:", labels_dic)
    label_names_dic         = dict((m[0][1][0], m[0][2][0]) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names             = [tup[1] for tup in sorted([(v,label_names_dic[k]) for k,v in labels_dic.items()], key=lambda x:x[0])]    
    fval_ground_truth       = open(devkit_path+'/data/ILSVRC2012_validation_ground_truth.txt','r')
    validation_ground_truth = [[int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
    fval_ground_truth.close()
    
    return labels_dic, label_names, validation_ground_truth


def parse_function(filename, label, image_size):
    # load and preprocess the image
    img_string = tf.io.read_file(filename)
    img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    img_resized = tf.image.resize(img_decoded, [224, 224])
    img_normalized = tf.keras.applications.resnet.preprocess_input(img_resized)
    
    # return the preprocessed image and its label
    return img_normalized

def read_data(prefix, labels_dic, mixing, files_from_cl):
    batch_size =128
    image_list = sorted(map(lambda x: os.path.join(prefix, x),
                        filter(lambda x: x.endswith('JPEG'), files_from_cl)))

    prefix2 = np.array([file_i.split(prefix + '/')[1].split("_")[0] for file_i in image_list])
    labels_list = np.array([mixing[labels_dic[i]] for i in prefix2])

    dataset = tf.data.Dataset.from_tensor_slices((image_list, labels_list))
    dataset = dataset.shuffle(buffer_size=len(image_list))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset


def read_data_test(data_path, labels_dic, mixing, files_from_cl=None, prefix='', image_size=224):
    '''Function to read randomly the training images'''
    nb_samples = 0
    image_list = []
    label_list = []
    file_string_list = []

    # Store each file in the list
    if files_from_cl is None:
        for key in labels_dic:
            temp_list = []
            path = os.path.join(data_path, key)
            files = glob.glob(path)
            np.random.shuffle(files)
            nb_files = len(files)
            nb_samples += nb_files
            temp_list.append(files)
            temp_list.append(labels_dic[key] * np.ones((nb_files,), dtype=np.int32))
            temp_list.append([key] * nb_files)
            image_list.append(temp_list[0])
            label_list.append(temp_list[1])
            file_string_list.append(temp_list[2])
    else:
        for i in range(len(files_from_cl)):
            temp_list = []
            label = files_from_cl[i].split('_')[0]
            if label in labels_dic:
                path = os.path.join(data_path, files_from_cl[i])
                files = glob.glob(path)
                np.random.shuffle(files)
                nb_files = len(files)
                nb_samples += nb_files
                temp_list.append(files)
                temp_list.append(mixing[labels_dic[label] * np.ones((nb_files,), dtype=np.int32)])
                temp_list.append([files_from_cl[i]] * nb_files)
                image_list.append(temp_list[0])
                label_list.append(temp_list[1])
                file_string_list.append(temp_list[2])
            else:
                print(f"Warning: Skipping {files_from_cl[i]} as it is not found in the labels dictionary.")
    image_list = np.hstack(image_list)
    label_list = np.hstack(label_list)
    file_string_list = np.hstack(file_string_list)

    # Random shuffle
    idx_shuffle = np.arange(nb_samples)
    np.random.shuffle(idx_shuffle)

    image_list = image_list[idx_shuffle]
    label_list = label_list[idx_shuffle]
    file_string_list = file_string_list[idx_shuffle]

    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list, file_string_list)).repeat()

    return dataset



    def parse_function(filename, label, file_string):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [image_size, image_size])
        image_normalized = tf.keras.applications.resnet.preprocess_input(image_resized)
        return image_normalized, label, file_string

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def prepare_files(train_path, mixing, order, labels_dic, nb_groups, nb_cl, nb_val):
    from collections import Counter
    files=os.listdir(train_path)
    
    prefix = np.array([file_i.split("_")[0] for file_i in files])
  #  print("Prefix array:", prefix)
  #  print("Length of prefix array:", len(prefix))
    
    missing_keys = [i for i in prefix if i not in labels_dic]
   # print("Missing keys:", len(missing_keys))
    missing_key_counts = Counter(missing_keys)
  #  print("Unique missing keys and their counts:", missing_key_counts)
    labels_old = np.array([mixing[labels_dic[i]] for i in prefix])
    
    files_train = []
    files_valid = []
    
    for _ in range(nb_groups):
      files_train.append([])
      files_valid.append([])
    
    files=np.array(files)
    
    for i in range(nb_groups):
      for i2 in range(nb_cl):
        tmp_ind=np.where(labels_old == order[nb_cl*i+i2])[0]
        np.random.shuffle(tmp_ind)
        files_train[i].extend(files[tmp_ind[0:len(tmp_ind)-nb_val]])
        files_valid[i].extend(files[tmp_ind[len(tmp_ind)-nb_val:]])
    
    return files_train, files_valid
