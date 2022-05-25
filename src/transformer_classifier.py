import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

"""
## Implement a Transformer block as a layer
"""
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer

Two seperate embedding layers, one for tokens, one for token index (positions).
"""
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


"""
## Implement a Transformer classifier
"""
class TransformerClassifier(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, transformer_blks=1, dense_dims=[64,], vocab_size=500, maxlen=200, y_maxlen=2, active='softmax'):
        self.embed_dim = embed_dim  # Embedding size for each token
        self.num_heads = num_heads  # Number of attention heads
        self.ff_dim = ff_dim  # Hidden layer size in feed forward network inside transformer
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.y_maxlen = y_maxlen
        self.activation = active
        self.dense_dims = dense_dims
        self.transformer_blks = transformer_blks

        self.embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        self.transformer_blocks_list = [TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim) for _ in range(self.transformer_blks)]

        input = layers.Input(shape=(self.maxlen,))
        x = self.embedding_layer(input)
        for blk in self.transformer_blocks_list:
            x = blk(x)
        # x = layers.Flatten()(x)
        x = layers.GlobalAveragePooling1D()(x)
        for dense_layer in self.dense_dims:
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(dense_layer, kernel_regularizer=regularizers.l2(0.005), activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.y_maxlen, activation=self.activation)(x)

        self.model = keras.Model(inputs=input, outputs=outputs)

    def compile(self, optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"]):
        self.model.summary()
        self.model.compile(optimizer, loss, metrics=metrics)

    def fit(self, xs, ys, valid_data, batch_size=16, epochs=5, verbose=1):
        return self.model.fit(xs, ys, batch_size=batch_size, epochs=epochs, validation_data=valid_data, verbose=verbose)

    def predict(self, xs):
        return self.model.predict(xs)

    def save_weights(self, path):
        return self.model.save_weights(path)

    def load_weights(self, path):
        return self.model.load_weights(path)


"""
## Implement an Augmented Transformer classifier
"""
class AugmentedTransformerClassifier(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, transformer_blks=1, dense_dims=[64,], vocab_size=500, maxlen=200, y_maxlen=2, next_tokens_len=59, active='softmax'):
        self.embed_dim = embed_dim  # Embedding size for each token
        self.num_heads = num_heads  # Number of attention heads
        self.ff_dim = ff_dim  # Hidden layer size in feed forward network inside transformer
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.y_maxlen = y_maxlen
        self.next_tokens_len = next_tokens_len
        self.activation = active
        self.dense_dims = dense_dims
        self.transformer_blks = transformer_blks

        self.embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        self.transformer_blocks_list = [TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim) for _ in range(self.transformer_blks)]

        seq_input = layers.Input(shape=(self.maxlen,))
        nexts_input = layers.Input(shape=(self.next_tokens_len,))
        x = self.embedding_layer(seq_input)
        for blk in self.transformer_blocks_list:
            x = blk(x)
        # x = layers.Flatten()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Concatenate()([x, nexts_input])
        for dense_layer in self.dense_dims:
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(dense_layer, kernel_regularizer=regularizers.l2(0.005), activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.y_maxlen, activation=self.activation)(x)

        self.model = keras.Model(inputs=[seq_input, nexts_input], outputs=outputs)

    def compile(self, optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"]):
        self.model.summary()
        self.model.compile(optimizer, loss, metrics=metrics)

    def fit(self, xs, ys, valid_data, batch_size=16, epochs=5, verbose=1):
        return self.model.fit(xs, ys, batch_size=batch_size, epochs=epochs, validation_data=valid_data, verbose=verbose)

    def predict(self, xs):
        return self.model.predict(xs)

    def save_weights(self, path):
        return self.model.save_weights(path)

    def load_weights(self, path):
        return self.model.load_weights(path)


if __name__ == "__main__":
    """
    ## Download and prepare dataset
    """
    vocab_size = 10000  # Only consider the top 10k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    """
    ## Create classifier model using transformer layer

    Transformer layer outputs one vector for each time step of our input sequence.
    Here, we take the mean across all time steps and
    use a feed forward network on top of it to classify text.
    """
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    transformerClfr = TransformerClassifier(embed_dim, num_heads, ff_dim, 1, [32], vocab_size, maxlen)

    """
    ## Train and Evaluate
    """
    transformerClfr.compile("adam", "sparse_categorical_crossentropy",
                metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy")])
    history = transformerClfr.fit(x_train, y_train, (x_val, y_val))
