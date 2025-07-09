import tensorflow as tf

class KAN(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation='relu', **kwargs):
        super(KAN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        # Univariate transformation: one small dense layer per input feature
        self.univariate_transforms = [
            tf.keras.layers.Dense(1, activation=self.activation) for _ in range(input_dim)
        ]

        # Final linear combination layer
        self.output_dense = tf.keras.layers.Dense(self.output_dim)

        super(KAN, self).build(input_shape)

    def call(self, inputs):
        # Apply univariate transforms independently
        transformed_features = []
        for i in range(inputs.shape[-1]):
            feature_column = tf.expand_dims(inputs[:, i], axis=-1)  # (batch_size, 1)
            transformed = self.univariate_transforms[i](feature_column)  # (batch_size, 1)
            transformed_features.append(transformed)

        # Concatenate all univariate outputs
        concat = tf.concat(transformed_features, axis=-1)  # (batch_size, input_dim)

        # Final linear layer
        output = self.output_dense(concat)  # (batch_size, output_dim)
        return output

    def get_config(self):
        config = super(KAN, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "activation": tf.keras.activations.serialize(self.activation)
        })
        return config
