import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

from forest_guard.params import FEATURES_DICT, FEATURES, BANDS, TRAINING_BASE, EVAL_BASE, BUCKET

def parse_tfrecord(example_proto):
    """The parsing function.
    Read a serialized example into the structure defined by FEATURES_DICT.
    Args:
    example_proto: a serialized Example.
    Returns:
    A dictionary of tensors, keyed by feature name.
    """
    return tf.io.parse_single_example(example_proto, FEATURES_DICT)


def to_tuple(inputs):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
    inputs: A dictionary of tensors, keyed by feature name.
    Returns:
    A tuple of (inputs, outputs).
    """
    inputsList = [inputs.get(key) for key in FEATURES]
    stacked = tf.stack(inputsList, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    return stacked[:,:,:len(BANDS)], stacked[:,:,len(BANDS):]


def get_dataset(pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
    pattern: A file pattern to match in a Cloud Storage bucket.
    Returns:
    A tf.data.Dataset
    """
    glob = tf.io.gfile.glob(pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(to_tuple, num_parallel_calls=5)
    return dataset

def target_simplification(x,y):
  y_simplified = tf.where(y>1., 0., y)
  return (x,y_simplified)

def get_training_dataset_gen(folder, batch_size=16, buffer_size = 2000):
    """Get the preprocessed training dataset
    Returns: 
    A tf.data.Dataset of training data.
    """
    glob = 'gs://' + BUCKET + '/' + folder + '/' + TRAINING_BASE + '*'
    dataset = get_dataset(glob)
    dataset = dataset.map(target_simplification)
    
    #Data augmentation
    
    data_gen_args = dict(horizontal_flip=True, vertical_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 9
    # import ipdb; ipdb.set_trace()
    feature = np.array([x for x, y in dataset])
    target = np.array([y for x, y in dataset])
    image_datagen.fit(feature, augment=True, seed=seed)
    mask_datagen.fit(target, augment=True, seed=seed)
    
    # # combine generators into one which yields image and masks
    image_datagen=image_datagen.flow(feature, batch_size=batch_size, seed=seed)
    mask_datagen=mask_datagen.flow(target, batch_size=batch_size, seed=seed)
    
    dataset = zip(image_datagen, mask_datagen)
    
    
    
    #transform in 
    #dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat()
    return dataset

def get_training_dataset(folder, batch_size=16, buffer_size = 2000):
    """Get the preprocessed training dataset
    Returns: 
    A tf.data.Dataset of training data.
    """
    glob = 'gs://' + BUCKET + '/' + folder + '/' + TRAINING_BASE + '*'
    dataset = get_dataset(glob)
    
    #transform in 
    dataset = dataset.shuffle(buffer_size).batch(batch_size).repeat()
    dataset = dataset.map(target_simplification).repeat()
    
    return dataset

def get_eval_dataset_gen(folder, batch_size=16, buffer_size = 2000):
    """Get the preprocessed eval dataset
    Returns: 
    A tf.data.Dataset of eval data.
    """
    glob = 'gs://' + BUCKET + '/' + folder + '/' + EVAL_BASE + '*'
    dataset = get_dataset(glob)
    dataset = dataset.map(target_simplification)
    
    #Data augmentation
    
    data_gen_args = dict(horizontal_flip=True, vertical_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 9
    # import ipdb; ipdb.set_trace()
    feature = np.array([x for x, y in dataset])
    target = np.array([y for x, y in dataset])
    image_datagen.fit(feature, augment=True, seed=seed)
    mask_datagen.fit(target, augment=True, seed=seed)

    # # combine generators into one which yields image and masks
    image_datagen=image_datagen.flow(feature, batch_size=1, seed=seed)
    mask_datagen=mask_datagen.flow(target, batch_size=1, seed=seed)
    
    dataset = zip(image_datagen, mask_datagen)
    
    
    # dataset = dataset.batch(1).repeat()
    return dataset


def get_eval_dataset(folder, batch_size=16, buffer_size = 2000):
    """Get the preprocessed eval dataset
    Returns: 
    A tf.data.Dataset of eval data.
    """
    glob = 'gs://' + BUCKET + '/' + folder + '/' + EVAL_BASE + '*'
    dataset = get_dataset(glob)
    dataset = dataset.batch(1).repeat()
    dataset = dataset.map(target_simplification).repeat()
    return dataset
