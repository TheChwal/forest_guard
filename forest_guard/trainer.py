import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from google.cloud import storage
import pickle
import pandas as pd

from forest_guard.params import BUCKET, FOLDER, BATCH_SIZE
from forest_guard.parse import get_training_dataset, get_eval_dataset

import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FR] [PARIS] [Forest] Forest Guard"
MODEL_STORAGE_LOCATION = 'models/forest_guard/'

class Trainer():
    def __init__(self, model_output_name, 
                 pretrained_google = True,
                 model_input_name = 'model_forest',
                 mlflow = True):
        self.experiment_name = EXPERIMENT_NAME
        self.mlflow = mlflow
        self.model_input_name = model_input_name
        self.model_output_name = model_output_name + '/'
        self.pretrained_google = pretrained_google


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    # def upload_model_to_gcp(self):
    #     client = storage.Client()
    #     bucket = client.bucket(BUCKET)
    #     blob = bucket.blob(MODEL_STORAGE_LOCATION+self.model_output_name)
    #     blob.upload_from_filename(self.model_output_name)
        
    def download_model_from_gcp(self, rm=True):
        if self.pretrained_google:
            MODEL_DIR = 'gs://ee-docs-demos/fcnn-demo/trainer/model'
        else:
            MODEL_DIR = 'gs://' + BUCKET + '/' + MODEL_STORAGE_LOCATION + self.model_input_name
        self.model = tf.keras.models.load_model(MODEL_DIR)

            # client = storage.Client().bucket(BUCKET)
            # storage_location = '{}/{}'.format(MODEL_STORAGE_LOCATION, self.model_input_name)
            # blob = client.blob(MODEL_STORAGE_LOCATION+self.model_input_name)
            # blob.download_to_filename('model.joblib')
            # print("=> model downloaded from storage")
            # self.model = joblib.load('model.joblib')
            # if rm:
            #     os.remove('model.joblib')
        return self
    
    def save_history(self, history):
        '''
        method to save the history of the fit
        '''
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = 'history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        
        client = storage.Client().bucket(BUCKET)
        storage_location = '{}{}history'.format(MODEL_STORAGE_LOCATION, 'history_'+self.model_output_name)
        blob = client.blob(storage_location)
        
        blob.upload_from_filename('history.csv')
        print("=> history save on cloud storage")
        
        #os.remove('history.csv')
        return None
    
    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        # file = self.model_output_name
        # joblib.dump(self.model, file)
        # print(f"saved {file} locally")

        # self.upload_model_to_gcp(file)
        # print(f"uploaded {file} to gcp cloud storage under \n => {MODEL_STORAGE_LOCATION+file}")
        
        MODEL_SAVE = 'gs://' + BUCKET + '/' + MODEL_STORAGE_LOCATION + self.model_output_name
        self.model.save(MODEL_SAVE)
        return None
    
    def run(self,
            training,
            evaluation,
            nb_epochs,
            train_size = 16000,
            eval_size = 8000,
            optimizer='SGD',
            loss='MeanSquaredError',
            metrics = ['RootMeanSquaredError'],
            patience = 5):

        """compile and fit the model  
        Return history
        """
        # write model in ml_flow
        if self.mlflow :
            self.mlflow_log_param('model', self.model_output_name) 
            self.mlflow_log_param('optimizer', optimizer) 
            self.mlflow_log_param('loss', loss) 



        self.model.compile(
                    optimizer=optimizers.get(optimizer), 
                    loss=losses.get(loss),
                    metrics=metrics)
        
        es = EarlyStopping(monitor='val_loss', mode='auto', patience=patience, verbose=1, restore_best_weights=True)

        history = self.model.fit(       x=training, 
                                        epochs=nb_epochs, 
                                        steps_per_epoch=int(train_size / BATCH_SIZE), 
                                        validation_data=evaluation,
                                        validation_steps=eval_size,
                                        callbacks=[es]  
                            )
        #history = model.fit(self.X, self.y)
        return history
    
    def metrics_to_mlflow(self, history):
        ''' write metrics in mlflow'''
        last_training_loss= history.history.get('loss', [0])[-1]
        last_val_loss=history.history.get('val_loss', [0])[-1]
        
        last_val_mean_io_u=history.history.get('val_mean_io_u',[0])[-1]
        if last_val_mean_io_u == 0:
            last_val_mean_io_u==history.history.get('val_mean_io_u_1',[0])[-1]
        
        last_val_accuracy=history.history.get('val_accuracy', [0])[-1]
        
        self.mlflow_log_metric('last_loss', last_training_loss)
        self.mlflow_log_metric('last_val_loss', last_val_loss)
        self.mlflow_log_metric('last_val_mean_io_u', last_val_mean_io_u)
        self.mlflow_log_metric('last_val_accuracy', last_val_accuracy)
        return None




if __name__ == "__main__":

    #get training and eval
    training = get_training_dataset(FOLDER)
    evaluation = get_eval_dataset(FOLDER)
    
    # hold out
    
    # train
    print('\n', 'instantiate trainer')
    trainer = Trainer('test_model')
    
    print('\n', 'download model')
    trainer.download_model_from_gcp()
    
    print('\n', 'run trainer')
    history = trainer.run(training, evaluation, 1, train_size = 160, eval_size=80)
    # write metrics
    trainer.metrics_to_mlflow(history)
    
    # save model
    trainer.save_history(history)
    print('\n', 'save model')
    trainer.save_model()
    pass
