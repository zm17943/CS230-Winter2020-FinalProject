from util import *
from functions import load_data
from functions import make_federated_data
from functions import preprocess
from functions import create_compiled_keras_model


# Parameters to tune
path = "/Users/zhangzhang/"          # Your path of h5 files       
NUM_CLIENTS = 4
NUM_EPOCHS = 100
BATCH_SIZE = 32
SHUFFLE_BUFFER = 500
NUM_AVERAGE_ROUND = 10


# Check the environment
warnings.simplefilter('ignore')
np.random.seed(0)
if six.PY3:
  tff.framework.set_default_executor(tff.framework.create_local_executor())
tff.federated_computation(lambda: 'The tensorflow federated environment is correctly setup!')()


# Load the data
emnist_train, emnist_test = load_data(path)



# Generate sample batch
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[1])
example_element = iter(example_dataset).next()
preprocessed_example_dataset = preprocess(example_dataset, NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE)
sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())



# Create federated data for each client
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients, NUM_EPOCHS, SHUFFLE_BUFFER, BATCH_SIZE)



# Function to create tff,learning instances
def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

model_fn()



# Build iterations
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()



# Begin training rounds
state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))
for round_num in range(2, NUM_AVERAGE_ROUND):
  round_train_data = make_federated_data(emnist_train, sample_clients)
  state, metrics = iterative_process.next(state, round_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))



# #Evaluation
# evaluation = tff.learning.build_federated_evaluation(model_fn)
# federated_test_data = make_federated_data(emnist_test, sample_clients)
# len(federated_test_data), federated_test_data[0]
# test_metrics = evaluation(state.model, federated_test_data)
# str(test_metrics)

