import tensorflow_federated as tff
import tensorflow as tf
from .vec_to_vec import build_ModelLink_vec2vec
import numpy as np
import attr
import nest_asyncio
nest_asyncio.apply()
import sys


def run_federated_mlink_vec2vec(input_len, output_len, dataset,
                                client_lr=0.01, server_lr=0.1, epochs=10):

    def model_fn():
        m = build_ModelLink_vec2vec(input_len, output_len)
        return tff.learning.from_keras_model(m,
                                             loss=tf.keras.losses.CategoricalCrossentropy(),
                                             input_spec=dataset[0].element_spec,
                                             metrics=[tf.keras.metrics.Accuracy()])
    
    training_process = tff.learning.algorithms.build_weighted_fed_avg(model_fn, 
                                                                      client_optimizer_fn=lambda: tf.keras.optimizers.RMSprop(learning_rate=client_lr),
                                                                      server_optimizer_fn=lambda: tf.keras.optimizers.RMSprop(learning_rate=server_lr))
    print(training_process.initialize.type_signature.formatted_representation())

    train_state = training_process.initialize()

    for round_n in range(epochs):
        result = training_process.next(train_state, dataset)
        train_state = result.state
        train_metrics = result.metrics
        print('round {:2d}, metrics={}'.format(round_n, train_metrics))


def run_federated_mlink_vec2vec_v2(input_len, output_len, dataset,
                                   client_lr=0.01, server_lr=0.1, epochs=10):

    def model_fn():
        m = build_ModelLink_vec2vec(input_len, output_len)
        return tff.learning.from_keras_model(m,
                                             loss=tf.keras.losses.CategoricalCrossentropy(),
                                             input_spec=dataset[0].element_spec,
                                             metrics=[tf.keras.metrics.Accuracy()])

    # https://www.tensorflow.org/federated/tutorials/custom_federated_algorithm_with_tff_optimizers
    @tf.function
    def client_update(model, dataset, server_weights, client_optimizer):
        client_weights = model.trainable_variables
        tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)
        trainable_tensor_specs = tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape, v.dtype), client_weights)
        optimizer_state = client_optimizer.initialize(trainable_tensor_specs)
        for batch in iter(dataset):
            with tf.GradientTape() as tape:
                outputs = model.forward_pass(batch)
            grads = tape.gradient(outputs.loss, client_weights)
            optimizer_state, updated_weights = client_optimizer.next(optimizer_state, client_weights, grads)
            tf.nest.map_structure(lambda a, b: a.assign(b), client_weights, updated_weights)
        res = tf.nest.map_structure(tf.subtract, client_weights, server_weights)
        print("CLIENT_UPDATE\n", res)
        print(f"SIZE={sys.getsizeof(res)} bytes")
        return res


    @attr.s(eq=False, frozen=True, slots=True)
    class ServerState(object):
        trainable_weights = attr.ib()
        optimizer_state = attr.ib()


    @tf.function
    def server_update(server_state, mean_model_delta, server_optimizer):
        """Updates the server model weights."""
        # Use aggregated negative model delta as pseudo gradient. 
        negative_weights_delta = tf.nest.map_structure(lambda w: -1.0 * w, mean_model_delta)
        new_optimizer_state, updated_weights = server_optimizer.next(server_state.optimizer_state, server_state.trainable_weights, negative_weights_delta)
        res = tff.structure.update_struct(server_state, trainable_weights=updated_weights, optimizer_state=new_optimizer_state)
        print("SERVER_UPDATED_WEIGHTS\n", updated_weights)
        print(f"SIZE={sys.getsizeof(updated_weights)} bytes")
        return res

    client_optimizer = tff.learning.optimizers.build_rmsprop(learning_rate=client_lr)
    server_optimizer = tff.learning.optimizers.build_rmsprop(learning_rate=server_lr)

    @tff.tf_computation
    def server_init():
        model = model_fn()
        trainable_tensor_specs = tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape, v.dtype), model.trainable_variables)
        optimizer_state = server_optimizer.initialize(trainable_tensor_specs)
        return ServerState(trainable_weights=model.trainable_variables, optimizer_state=optimizer_state)
    
    @tff.federated_computation
    def server_init_tff():
        return tff.federated_value(server_init(), tff.SERVER)
    
    server_state_type = server_init.type_signature.result
    # print('\nserver_state_type:\n', server_state_type.formatted_representation())
    trainable_weights_type = server_state_type.trainable_weights
    # print('\ntrainable_weights_type:\n', trainable_weights_type.formatted_representation())

    @tff.tf_computation(server_state_type, trainable_weights_type)
    def server_update_fn(server_state, model_delta):
        return server_update(server_state, model_delta, server_optimizer)
    
    model = model_fn()
    tf_dataset_type = tff.SequenceType(model.input_spec)
    # # print('\ntf_dataset_type:\n', tf_dataset_type.formatted_representation())

    # NOTE 是这一段代码输出了 CLIENT_UPDATE
    @tff.tf_computation(tf_dataset_type, trainable_weights_type)
    def client_update_fn(dataset, server_weights):
        model = model_fn()
        return client_update(model, dataset, server_weights, client_optimizer)
    
    federated_server_type = tff.FederatedType(server_state_type, tff.SERVER)
    federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

    @tff.federated_computation(federated_server_type, federated_dataset_type)
    def run_one_round(server_state, federated_dataset):
        # Server-to-client broadcast.
        server_weights_at_client = tff.federated_broadcast(server_state.trainable_weights)
        # Local client update.
        model_deltas = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))
        # Client-to-server upload and aggregation.
        mean_model_delta = tff.federated_mean(model_deltas)
        # Server update.
        server_state = tff.federated_map(server_update_fn, (server_state, mean_model_delta))
        return server_state
    
    fedavg_process = tff.templates.IterativeProcess(initialize_fn=server_init_tff, next_fn=run_one_round)
    # print('\ntype signature of `initialize`:\n', fedavg_process.initialize.type_signature.formatted_representation())
    # print('\ntype signature of `next`:\n', fedavg_process.next.type_signature.formatted_representation())

    server_state = fedavg_process.initialize()

    for round_n in range(epochs):
        server_state = fedavg_process.next(server_state, dataset)
        print('round {:2d}'.format(round_n))


# debug
if __name__ == "__main__":
    # dataset
    # 2 clients
    x1 = np.random.rand(100, 16)
    y1 = np.random.rand(100, 16)
    d1 = tf.data.Dataset.from_tensor_slices((x1, y1)).batch(32)

    x2 = np.random.rand(100, 16)
    y2 = np.random.rand(100, 16)
    d2 = tf.data.Dataset.from_tensor_slices((x2, y2)).batch(32)

    fed_d = [d1, d2]

    run_federated_mlink_vec2vec_v2(16, 16, fed_d, epochs=5)