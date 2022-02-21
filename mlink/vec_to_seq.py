import tensorflow as tf 
import numpy as np 
import os 
import time


class VecEncoder(tf.keras.Model):
    def __init__(self, enc_units):
        super(VecEncoder, self).__init__()
        self.enc_units = enc_units
        self.dense = tf.keras.layers.Dense(enc_units, activation="relu")
    
    def call(self, x):
        x = self.dense(x)
        return x


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    hidden_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class SeqDecoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(SeqDecoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))

    x = self.fc(output)

    return x, state, attention_weights


def build_ModelLink_vec2seq(output_dim, batch_size):
    encoder = VecEncoder(output_dim*2)
    decoder = SeqDecoder(output_dim, 64, 64, batch_size)
    return encoder, decoder


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(encoder, decoder, inp, targ, targ_lang_word_index, BATCH_SIZE, loss_object, optimizer):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output = encoder(inp)
    dec_hidden = enc_output
    dec_input = tf.expand_dims([targ_lang_word_index['<start>']] * BATCH_SIZE, 1)
    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions, loss_object)
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


def train_ModelLink_vec2seq(encoder, decoder, X, Y,
                            targ_lang_word_index,
                            learning_rate=0.01,
                            batch_size=32,
                            epochs=100, weight_dir="weights/"):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    checkpoint_dir = weight_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    
    EPOCHS = epochs
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.batch(batch_size)
    steps_per_epoch = len(X)

    for epoch in range(EPOCHS):
        start = time.time()

        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(encoder, decoder, inp, targ, targ_lang_word_index, batch_size,
                                    loss_object, optimizer)
            total_loss += batch_loss

            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))