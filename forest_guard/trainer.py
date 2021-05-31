import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanIoU
from tensorflow.python.keras import layers


from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
from google.cloud import storage
import pandas as pd
from tensorflow.python.ops.gen_math_ops import sparse_segment_mean_with_num_segments_eager_fallback

from forest_guard.params import BUCKET, FOLDER, BATCH_SIZE, MODEL_STORAGE_LOCATION, PROJECT, BANDS
from forest_guard.parse import get_training_dataset, get_eval_dataset
from forest_guard.losses import dice_loss, tversky_loss, lovasz_softmax, iou

import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FR] [PARIS] [Forest] Forest Guard"

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

    def download_model_from_gcp(self, rm=True):
        if self.pretrained_google:
            MODEL_DIR = 'gs://ee-docs-demos/fcnn-demo/trainer/model'
        else:
            MODEL_DIR = 'gs://' + BUCKET + '/' + MODEL_STORAGE_LOCATION + self.model_input_name
        self.model = tf.keras.models.load_model(MODEL_DIR)

        return self
    
    def save_history(self, history):
        '''
        method to save the history of the fit
        '''
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = 'history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        
        client = storage.Client(project=PROJECT).bucket(BUCKET)
        storage_location = '{}{}history'.format(MODEL_STORAGE_LOCATION, 'history_'+self.model_output_name)
        blob = client.blob(storage_location)
        
        blob.upload_from_filename('history.csv')
        print("=> history save on cloud storage")
        
        #os.remove('history.csv')
        return None
   #######################################"
   #### MODEL FROM SCRATCH
   #######################################
    def conv_block(self, input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder
    
    def encoder_block(self, input_tensor, num_filters):
        encoder = self.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
        return encoder_pool, encoder
    
    def decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder
    
    def init_model(self):    
        inputs = layers.Input(shape=[256, 256, len(BANDS)]) # 256
        encoder0_pool, encoder0 = self.encoder_block(inputs, 32) # 128
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64) # 64
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128) # 32
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256) # 16
        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512) # 8
        center = self.conv_block(encoder4_pool, 1024) # center
        decoder4 = self.decoder_block(center, encoder4, 512) # 16
        decoder3 = self.decoder_block(decoder4, encoder3, 256) # 32
        decoder2 = self.decoder_block(decoder3, encoder2, 128) # 64
        decoder1 = self.decoder_block(decoder2, encoder1, 64) # 128
        decoder0 = self.decoder_block(decoder1, encoder0, 32) # 256
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
        self.model = models.Model(inputs=[inputs], outputs=[outputs])
        return self
    
    ######################################" END MODEL FROM SCRATCH"
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
            optimizer='adam',
            loss='binary_crossentropy',
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
    ################
    ## UPDATE FOLDER
    ################
    training = get_training_dataset(FOLDER)
    evaluation = get_eval_dataset(FOLDER)
    
    # train
    print('\n', 'instantiate trainer')
    ################
    ## UPDATE NAME
    ################
    trainer = Trainer('ai_platform_tversky_ownmodel')
    
    # print('\n', 'download model')
    # trainer.download_model_from_gcp()
    
    trainer.init_model()
    
    #iou = MeanIoU(num_classes=2)
    print('\n', 'run trainer')
    history = trainer.run(training,
                      evaluation,
                      100,
                      metrics = [iou, "mae", "accuracy"], 
                      optimizer='adam',
                      loss=tversky_loss(0.5) ,
                      train_size = 1600,
                      eval_size=800,
                     patience=10)
    # write metrics
    trainer.metrics_to_mlflow(history)
    
    # save model
    print('\n', 'save model')
    trainer.save_model()
    
    #save_history
    trainer.save_history(history)
    pass
