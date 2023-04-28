import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import scipy
import os
import scipy.io
import sys
try:
    import cPickle
except:
    import _pickle as cPickle
# Syspath for the folder with the utils files
#sys.path.insert(0, "/media/data/srebuffi")

import utils_resnet
import utils_icarl
import utils_data

######### Modifiable Settings ##########
batch_size = 64          # Batch size
nb_val     = 50             # Validation samples per class
nb_cl      = 10         # Classes per group 
nb_groups  = 100          # Number of groups
nb_proto   = 20             # Number of prototypes per class: total protoset memory/ total number of classes
epochs     = 64             # Total number of epochs 
lr_old     = 2.             # Initial learning rate
lr_strat   = [20,30,40,50]  # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
gpu        = '0'            # Used GPU
wght_decay = 0.00001        # Weight Decay
########################################

######### Paths  ##########
# Working station 
devkit_path = 'meta'
train_path  = 'ILSVRC/Data/CLS-LOC/new_train'
save_path   = 'save_path'

###########################

#####################################################################################################

### Initialization of some variables ###
class_means    = np.zeros((512,nb_groups*nb_cl,2,nb_groups))
loss_batch     = []
files_protoset =[]
for _ in range(nb_groups*nb_cl):
    files_protoset.append([])


### Preparing the files for the training/validation ###

# Random mixing
print("Mixing the classes and putting them in batches of classes...")
np.random.seed(1993)
order  = np.arange(nb_groups * nb_cl)
mixing = np.arange(nb_groups * nb_cl)
np.random.shuffle(mixing)

# Loading the labels
labels_dic, label_names, validation_ground_truth = utils_data.parse_devkit_meta(devkit_path)
# Or you can just do like this
# define_class = ['apple', 'banana', 'cat', 'dog', 'elephant', 'forg']
# labels_dic = {k: v for v, k in enumerate(define_class)}

# Preparing the files per group of classes
print("Creating a validation set ...")
files_train, files_valid = utils_data.prepare_files(train_path, mixing, order, labels_dic, nb_groups, nb_cl, nb_val)

# Pickle order and files lists and mixing
with open(str(nb_cl)+'mixing.pickle','wb') as fp:
    cPickle.dump(mixing,fp)

with open(str(nb_cl)+'settings_resnet.pickle','wb') as fp:
    cPickle.dump(order,fp)
    cPickle.dump(files_valid,fp)
    cPickle.dump(files_train,fp)


### Start of the main algorithm ###

for itera in range(nb_groups):
  
  # Files to load : training samples + protoset
  print('Batch of classes number {0} arrives ...'.format(itera+1))
  # Adding the stored exemplars to the training set
  if itera == 0:
    files_from_cl = files_train[itera]
  else:
    files_from_cl = files_train[itera][:]
    for i in range(itera*nb_cl):
      nb_protos_cl  = int(np.ceil(nb_proto*nb_groups*1./itera))
    
    ## Exemplars management part  ##
nb_protos_cl  = int(np.ceil(nb_proto*nb_groups*1./(itera+1))) # Reducing number of exemplars for the previous classes
files_from_cl = files_train[itera]
inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(files_from_cl, gpu, itera, batch_size, train_path, labels_dic, mixing, nb_groups, nb_cl, save_path)

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def read_and_preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    print("here")
    img = tf.keras.applications.resnet.preprocess_input(img)
    return img

# create a dataset from the list of filenames
files_train_tf = tf.data.Dataset.from_tensor_slices(files_from_cl)

# map the parse_function to the dataset, passing label and mixing as additional arguments
dataset = files_train_tf.map(lambda x: (parse_function(x), labels_dic[x.split('/')[0]]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

# convert labels to one-hot encoding
label_train_tf = np.array([labels_dic[file.split('/')[0]] for file in files_from_cl])
label_train_tf = tf.one_hot(label_train_tf, nb_groups * nb_cl)

# create batched dataset
train_ds = tf.data.Dataset.zip((image_train_tf, label_train_tf))
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

for epoch in range(epochs):
    print("Batch of classes {} out of {} batches".format(itera + 1, nb_groups))
    print('Epoch %i' % epoch)
    for image_batch_tf, label_batch_tf in train_ds:
        loss_class_val, _, sc = sess.run([loss_class, train_step, scores], feed_dict={learning_rate: lr, image_train: image_batch_tf, label_train: label_batch_tf})
        loss_batch.append(loss_class_val)

        # Plot the training error every 10 batches
        if len(loss_batch) == 10:
            print(np.mean(loss_batch))
            loss_batch = []

        # Plot the training top 1 accuracy every 80 batches
        if (i + 1) % 80 == 0:
            stat = []
            stat += ([ll in best for ll, best in zip(label_batch, np.argsort(sc, axis=1)[:, -1:])])
            stat = np.average(stat)
            print('Training accuracy %f' % stat)

    # Decrease the learning by 5 every 10 epoch after 20 epochs at the first learning rate
    if epoch in lr_strat:
        lr /= lr_factor

coord.request_stop()
coord.join(threads)

# copy weights to store network
save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])
utils_resnet.save_model(save_path+'model-iteration'+str(nb_cl)+'-%i.pickle' % itera, scope='ResNet18', sess=sess)

# Reset the graph
tf.reset_default_graph()
