import tensorflow as tf

class CoLSTM(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = 64
        self.output_size = 1
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        # Convolution Hyperparameters
        self.kernel_size = (3, 3) 
        self.num_filters = 7 
        self.pool_size = (2, 2) 
        self.strides = (1, 1) 
        self.padding = "same" 

        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)
        self.cnn = tf.keras.layers.Conv2D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size)
        self.lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
    
    @tf.function
    def call(self, reviews, training=True):
        review_embeddings = self.embedding(reviews)
        review_embeddings = tf.expand_dims(review_embeddings, axis=-1)

        cnn_output = self.cnn(review_embeddings)
        cnn_normalized = self.batch_norm(cnn_output, training=training)
        dropout = self.dropout(cnn_normalized, training=training)
        cnn_pooled = self.pooling(dropout)
        cnn_pooled = tf.reshape(cnn_pooled, [tf.shape(cnn_pooled)[0], tf.shape(cnn_pooled)[1], -1])

        lstm_out = self.lstm(cnn_pooled)

        dense_out = self.dense(lstm_out[0])
        return dense_out
    
    def train(self, reviews, labels, batch_size=30):
        shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
        reviews_shuffled = tf.gather(reviews, shuffled_indices)
        labels_shuffled = tf.gather(labels, shuffled_indices)

        losses = []
        accuracies = []
        for _, end in enumerate(range(batch_size, len(reviews_shuffled) + 1, batch_size)):
            start = end - batch_size
            train_inputs_batches = reviews_shuffled[start:end]
            train_labels_batches = labels_shuffled[start:end]

            with tf.GradientTape() as tape:
                probs = self(train_inputs_batches)
                probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1)
                predictions = tf.cast(probs >= 0.5, tf.float32)
                true_labels = tf.cast(train_labels_batches, tf.float32)
                loss = tf.keras.losses.binary_crossentropy(true_labels, probs)
                losses.append(loss)
                accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
                accuracies.append(accuracy)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return tf.math.reduce_mean(losses), tf.math.reduce_mean(accuracies)

    def test(self, reviews, labels):
        probs = self(reviews, training=False)
        probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1)
        loss = tf.keras.losses.binary_crossentropy(labels, probs)
        predictions = tf.cast(probs >= 0.5, tf.float32)
        true_labels = tf.cast(labels, tf.float32)
        accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
        return loss, accuracy
    

#####################################################################
### LSTM-Only Model
class LSTM(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = 64
        self.output_size = 1
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
    
    @tf.function
    def call(self, reviews, training=True):
        review_embeddings = self.embedding(reviews)

        lstm_out = self.lstm(review_embeddings)
        lstm_normalized = self.batch_norm(lstm_out[0], training=training)
        lstm_dropout = self.dropout(lstm_normalized, training=training)

        dense_out = self.dense(lstm_dropout)
        return dense_out

    def train(self, reviews, labels, batch_size=30):
        shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
        reviews_shuffled = tf.gather(reviews, shuffled_indices)
        labels_shuffled = tf.gather(labels, shuffled_indices)

        losses = []
        accuracies = []
        for _, end in enumerate(range(batch_size, len(reviews_shuffled)+1, batch_size)):
            start = end - batch_size
            train_inputs_batches = reviews_shuffled[start:end]
            train_labels_batches = labels_shuffled[start:end]

            with tf.GradientTape() as tape:
                probs = self(train_inputs_batches)
                probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1)
                predictions = tf.cast(probs >= 0.5, tf.float32)
                true_labels = tf.cast(train_labels_batches, tf.float32)
                loss = tf.keras.losses.binary_crossentropy(true_labels, probs)
                losses.append(loss)
                accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
                accuracies.append(accuracy)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return tf.math.reduce_mean(losses), tf.math.reduce_mean(accuracies)

    def test(self, reviews, labels):
        probs = self(reviews, training=False)
        probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1)
        loss = tf.keras.losses.binary_crossentropy(labels, probs)
        predictions = tf.cast(probs >= 0.5, tf.float32)
        true_labels = tf.cast(labels, tf.float32)
        accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
        return loss, accuracy
    
##################################################
### CNN-Only Model
class CNN(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = 64
        self.output_size = 1
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        self.kernel_size = (3, 3)
        self.num_filters = 7
        self.pool_size = (2, 2)
        self.strides = (1, 1)
        self.padding = "same"

        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)
        self.cnn = tf.keras.layers.Conv2D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size)
        self.dense = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")

    @tf.function
    def call(self, reviews, training=True):
        review_embeddings = self.embedding(reviews)
        review_embeddings = tf.expand_dims(review_embeddings, axis=-1)

        cnn_output = self.cnn(review_embeddings)
        cnn_normalized = self.batch_norm(cnn_output, training=training)
        dropout = self.dropout(cnn_normalized, training=training)
        cnn_pooled = self.pooling(dropout)
        cnn_pooled = tf.reshape(cnn_pooled, [tf.shape(cnn_pooled)[0], tf.shape(cnn_pooled)[1], -1])

        dense_out = self.dense(cnn_pooled)
        return dense_out

    def train(self, reviews, labels, batch_size=30):
        shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
        reviews_shuffled = tf.gather(reviews, shuffled_indices)
        labels_shuffled = tf.gather(labels, shuffled_indices)

        losses = []
        accuracies = []
        for _, end in enumerate(range(batch_size, len(reviews_shuffled)+1, batch_size)):
            start = end - batch_size
            train_inputs_batches = reviews_shuffled[start:end]
            train_labels_batches = labels_shuffled[start:end]

            with tf.GradientTape() as tape:
                probs = self(train_inputs_batches)
                probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1)
                predictions = tf.cast(probs >= 0.5, tf.float32)
                true_labels = tf.cast(train_labels_batches, tf.float32)
                loss = tf.keras.losses.binary_crossentropy(true_labels, probs)
                losses.append(loss)
                accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
                accuracies.append(accuracy)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return tf.math.reduce_mean(losses), tf.math.reduce_mean(accuracies)

    def test(self, reviews, labels):
        probs = self(reviews, training=False)
        probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1)
        loss = tf.keras.losses.binary_crossentropy(labels, probs)
        predictions = tf.cast(probs >= 0.5, tf.float32)
        true_labels = tf.cast(labels, tf.float32)
        accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
        return loss, accuracy