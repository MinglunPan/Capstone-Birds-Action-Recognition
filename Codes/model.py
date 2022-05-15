import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
import config
import os

@tf.function
def accuracy(y_true, y_pred):
    y_pred = tf.nn.sigmoid(y_pred)
    y_pred = tf.where(y_pred < 0.5, 0.0, 1.0)
    correct_count = tf_count(y_true, y_pred)
    total_count = keras.backend.shape(y_true)[0]
    return correct_count/total_count

@tf.function
def tf_count(t, val):
    elements_equal_to_value = keras.backend.equal(t, val)
    as_ints = keras.backend.cast(elements_equal_to_value, tf.int32)
    count = keras.backend.sum(as_ints,0)
    return count


def build_model_i3d(hub_url = "../pretrained-models/i3d-kinetics-400_1", num_output = config.NUM_ACT_CAT):
    hub_layer=hub.KerasLayer(hub.load(hub_url).signatures['default'], trainable=True)

    i3d_model = keras.Sequential()
    i3d_model.add(keras.Input(shape=config.X_SHAPE))
    i3d_model.add(hub_layer)
    i3d_model.add(keras.layers.Dense(num_output, activation = 'softmax'))
    return i3d_model

def build_model_i3d_dropout(hub_url = "../pretrained-models/i3d-kinetics-400_1", num_output = config.NUM_ACT_CAT):
    hub_layer=hub.KerasLayer(hub.load(hub_url).signatures['default'], trainable=True)

    i3d_model = keras.Sequential()
    i3d_model.add(keras.Input(shape=config.X_SHAPE))
    i3d_model.add(hub_layer)
    i3d_model.add(keras.layers.AlphaDropout(rate=0.5))
    i3d_model.add(keras.layers.Dense(num_output, activation = 'softmax'))
    return i3d_model

def build_model_old():
    #input_1, tracks
    track_input = keras.Input(shape=(config.NUM_INPUT_FRAME, config.IMG_HEIGHT, 
                                     config.IMG_WIDTH, config.NUM_CHANNEL), name="input_1")  # (?, 40, 200, 200, 3)

    #input_2, the 1/0 list, identify the real frames with 0, padding frames with 0
    list_input = keras.Input(shape=(config.NUM_INPUT_FRAME, 1, 1, 1), name="input_2")  # (?, 40, 1, 1, 1)

    x = track_input # (?, 40, 200, 200, 3)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, 3, padding='same'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, padding='same'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, padding='same'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x) # (?, 40, ?, ?, 32)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, padding='same'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x) # (?, 40, ?, ?, 32)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, padding='same'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.LayerNormalization())(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x) # (?, 40, ?, ?, 32)

    # attention layer
    attention_score = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x) # (?, 40, 32)
    attention_score = tf.keras.layers.Reshape((40, 1, 1, 32))(attention_score) # ?, 40, 1, 1, 32
    attention_score = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(4, 1, padding='same', activation='relu'))(attention_score) # ?, 40, 1, 1, 4
    attention_score = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 1, padding='same'))(attention_score) # ?, 40, 1, 1, 32
    attention_score = tf.keras.activations.sigmoid(attention_score) # ?, 40, 1, 1, 32

    x = x * attention_score  # (?, 40, ?, ?, 32)

    x = x * list_input # (batch_size, 40, ?, ?, 32)
    x = tf.reduce_sum(x, 1) # (batch_size, ?, ?, 32)
    x = x / tf.reduce_sum(list_input, 1) # (batch_size, ?, ?, 32)
    x = tf.keras.layers.Flatten()(x)
    Dense_LN_1 = tf.keras.layers.LayerNormalization()
    x = Dense_LN_1(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16)(x)
    Dense_LN_2 = tf.keras.layers.LayerNormalization()
    x = Dense_LN_2(x)
    x = tf.keras.activations.relu(x)
    class_pred = tf.keras.layers.Dense(1, name="bnb")(x)
    model = keras.Model(
        inputs=[track_input, list_input],
        outputs=[class_pred],
    )
    return model