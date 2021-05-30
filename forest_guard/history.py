'''
module to:
- retrieve history from cloud storage
- provide the plot function
'''
from forest_guard.params import BUCKET, MODEL_STORAGE_LOCATION
from google.cloud import storage
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_history(model_name):
    '''
    method to get back the history of the fit
    and return the dict
    '''
    hist_csv_file = 'history.csv'
    
    
    client = storage.Client().bucket(BUCKET)
    storage_location = '{}{}/history'.format(MODEL_STORAGE_LOCATION, 'history_'+model_name)
    blob = client.blob(storage_location)
    
    blob.download_to_filename(hist_csv_file)
    print("=> history loaded from cloud storage")
    history = pd.read_csv(hist_csv_file)
    
    os.remove(hist_csv_file)
    return history

def get_history_colab(model_name):
    '''
    method to get back the history of the fit
    and return the dict
    '''
    
    model_name_storage = model_name+'/'
    storage_location = '{}{}history'.format(MODEL_STORAGE_LOCATION, 'history_'+model_name_storage)

    hist_csv_file = 'gs://'+BUCKET+'/'+storage_location
    hist = pd.read_csv(hist_csv_file)

    return hist
    

def plot_history_accuracy(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history['loss'], label='train' + exp_name)
    ax1.plot(history['val_loss'], label='val' + exp_name)
    #ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)
