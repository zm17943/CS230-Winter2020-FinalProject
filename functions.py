from util import *
# from main import NUM_CLIENTS
# from main import NUM_EPOCHS
# from main import BATCH_SIZE
# from main import SHUFFLE_BUFFER

seed = 7
np.random.seed(seed)


def load_data(path, only_digits=True, cache_dir=None):
  """Loads the Federated customized dataset.
  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.ClientData` objects.
  """
  
  train_client_data = hdf5_client_data.HDF5ClientData(path + 'train.h5')
  test_client_data = hdf5_client_data.HDF5ClientData(path + 'test.h5')
  return train_client_data, test_client_data



def preprocess(dataset, NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE):

  def element_fn(element):
    return collections.OrderedDict([
        ('x', element['pixels']),
        ('y', tf.reshape(element['label'], [1])),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)



def make_federated_data(client_data, client_ids, NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE):
  return [preprocess(client_data.create_tf_dataset_for_client(x), NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE)
          for x in client_ids]



 
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = tf.keras.layers.Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=3,name=bn_name)(x)
    return x
 
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = tf.keras.layers.add([x,shortcut])
        return x
    else:
        x = tf.keras.layers.add([x,inpt])
        return x

def create_compiled_keras_model():
  inputs = tf.keras.layers.Input(shape=(256, 256, 3, ))

  x = tf.keras.layers.ZeroPadding2D((3,3))(inputs)

  x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
  x = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
  #(56,56,64)
  x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
  #(28,28,128)
  x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
  x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
  #(14,14,256)
  x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
  x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
  #(7,7,512)
  x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
  x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
  x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
  x = tf.keras.layers.AveragePooling2D(pool_size=(7,7))(x)

  x = tf.keras.layers.Dense(2, activation='softmax')(x)
  x = tf.keras.layers.Flatten()(x)

  model = tf.keras.models.Model(inputs=inputs, outputs=x)
  
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model



def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)