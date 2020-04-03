import tensorflow as tf
import datetime

class ModelLoader:
    def load_model(self, filename):
        return tf.keras.models.load_model('models/' + filename)

    def save_model(self, model, filename):
        #now = datetime.datetime.now()
        #now_string = now.strftime("%Y-%m-%dT%H:%M")
        model.save('models/' + filename + '.h5')