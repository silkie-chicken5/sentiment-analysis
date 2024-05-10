import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

class CoLSTM(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        print("in init")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # self.window_size = window_size
        self.embed_size = 64
        self.output_size = 1 # pos or neg
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        # Convolution Hyperparameters  CNN MIGHT BE STRIDE 1 VALID PADDING?
        # self.kernel_size = (3,3)
        # self.kernel_size = 3
        # self.num_filters = 7
        # # self.pool_size = (2, 2)
        # self.pool_size = 2
        # self.strides = None # defaults to pool size
        # self.padding = "same"
        
        # self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)
        # self.cnn = tf.keras.layers.Conv1D(
        #     filters = self.num_filters, 
        #     kernel_size=self.kernel_size, 
        #     padding=self.padding, # strides is default 1
        # )

        # self.pooling = tf.keras.layers.MaxPooling1D(
        #     pool_size=self.pool_size,
        #     strides=self.strides,
        #     padding=self.padding,
        # )

        #2D
        self.kernel_size = (3, 3) # Now specifying height and width for Conv2D
        self.num_filters = 7 # 3 # used to be 7
        self.pool_size = (2, 2) # For 2D pooling
        self.strides = (1, 1) # Stride of 1
        self.padding = "same" # This will reduce the dimension as no padding is added

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
        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
        # self.dense1 = tf.keras.layers.Dense(units=self.output_size, activation="softmax")
        # self.dense2 = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
    
    @tf.function
    def call(self, reviews, training=True):
        # print("in call")
        review_embeddings = self.embedding(reviews) # reviews need to have dimension of self.vocab_size
        # print("review embeddings shape: ", review_embeddings.shape)

        # cnn_output = self.cnn(review_embeddings)
        # print("cnn_output shape: ", cnn_output.shape)
        # cnn_pooled = self.pooling(cnn_output)
        # print("cnn pooled shape: ", cnn_pooled)
        
        review_embeddings = tf.expand_dims(review_embeddings, axis=-1)
        # print("review embeddings shape expanded: ", review_embeddings.shape)

        cnn_output = self.cnn(review_embeddings)
        cnn_normalized = self.batch_norm(cnn_output, training=training)
        dropout = self.dropout(cnn_normalized, training=training)
        cnn_pooled = self.pooling(dropout)

        # cnn_output = tf.squeeze(cnn_output, axis=2)
        # print("cnn_output shape: ", cnn_output.shape)

        # cnn_pooled = self.pooling(cnn_normalized)

        # cnn_pooled = self.pooling(cnn_output)
        # cnn_pooled = tf.reshape(cnn_pooled, [tf.shape(cnn_pooled)[0], -1, self.num_filters])

        cnn_pooled = tf.reshape(cnn_pooled, [tf.shape(cnn_pooled)[0], tf.shape(cnn_pooled)[1], -1])
        # print("cnn pooled shape: ", cnn_pooled)
        # pool_normalized = self.batch_norm(cnn_pooled, training=training)
        # pool_dropout = self.dropout(pool_normalized, training=training)
        # pool_dropout = tf.reshape(pool_dropout, [tf.shape(pool_dropout)[0], tf.shape(pool_dropout)[1], -1])


        lstm_out = self.lstm(cnn_pooled)
        # lstm_out = self.lstm(pool_dropout)
        # lstm_out = self.lstm(review_embeddings)
        # test = tf.expand_dims(lstm_out[0], axis=-1)
        # lstm_normalized = self.batch_norm(test, training=training)
        # lstm_dropout = self.dropout(lstm_normalized, training=training)

        # print("lstm out shape: ", lstm_out[0].shape)
        # dropout = self.dropout(lstm_out[0]) # Daniel: added dropout layer
        # print("dropout shape: ", dropout)
        # dense_out = self.dense(dropout)
        dense_out = self.dense(lstm_out[0])
        # dense_out1 = self.dense1(lstm_out[0])
        # dense_out = self.dense2(dense_out1)
        # dense_out = self.dense(lstm_dropout)
        # print("dense out shape is: ", dense_out.shape)
        # print("output size is: ", self.output_size)
        return dense_out
        return tf.nn.softmax(dense_out) # MIGHT NOT BE NECESSARY TO SOFTMAX BECAUSE SIGMOID ACTIVATION ALREADY RETURNS PROBS

#     def compile(self, optimizer, loss, metrics):
#         '''
#         Create a facade to mimic normal keras fit routine
#         '''
#         self.optimizer = optimizer
#         self.loss_function = loss 
#         self.accuracy_function = metrics[0]

    # def train(self, reviews, labels, batch_size=30):
    #     """
    #     Runs through one epoch - all training examples.

    #     :param model: the initialized model to use for forward and backward pass
    #     :param train_captions: train data captions (all data for training) 
    #     :param train_images: train image features (all data for training) 
    #     :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    #     :return: None
    #     """

    #     shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
    #     reviews_shuffled = tf.gather(reviews, shuffled_indices)
    #     labels_shuffled = tf.gather(labels, shuffled_indices)

    #     total_loss = 0
    #     # for start in range(0, len(reviews_shuffled), batch_size):
    #     for index, end in enumerate(range(batch_size, len(reviews_shuffled)+1, batch_size)):
    #         start = end - batch_size
    #         # end = min(start + batch_size, len(reviews_shuffled))
    #         train_inputs_batches = reviews_shuffled[start:end]
    #         train_labels_batches = labels_shuffled[start:end]

    #         with tf.GradientTape() as tape:
    #             probs = self(train_inputs_batches)
    #             loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs)
    #             # print("loss shape is: ", tf.shape(loss))
        
    #         gradients = tape.gradient(loss, self.trainable_variables)
    #         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    #         total_loss += loss
    #         # print("total loss shape is: ", tf.shape(total_loss))

    #     avg_loss = total_loss / (len(reviews_shuffled) / batch_size)
    #     return avg_loss
    
    def train(self, reviews, labels, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_images: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """

        ## TODO: Implement similar to test below.

        ## NOTE: shuffle the training examples (perhaps using tf.random.shuffle on a
        ##       range of indices spanning # of training entries, then tf.gather) 
        ##       to make training smoother over multiple epochs.

        ## NOTE: make sure you are calculating gradients and optimizing as appropriate
        ##       (similar to batch_step from HW2)
        print("in coLSTM train")
        # num_batches = int(len(reviews) / batch_size)
        shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
        reviews_shuffled = tf.gather(reviews, shuffled_indices)
        labels_shuffled = tf.gather(labels, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        losses = []
        accuracies = []
        for index, end in enumerate(range(batch_size, len(reviews_shuffled)+1, batch_size)):
            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            # print("TRAINING: ", start)
            # b0 = end - batch_size
            # train_inputs_batches = reviews_shuffled[b0:end]
            # train_labels_batches = labels_shuffled[b0:end]
            train_inputs_batches = reviews_shuffled[start:end]
            train_labels_batches = labels_shuffled[start:end]
            # batch_image_features = image_features_shuffled[start:end, :]
            # decoder_input = captions_shuffled[start:end, :-1]
            # decoder_labels = captions_shuffled[start:end, 1:]

            with tf.GradientTape() as tape:
                ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
                probs = self(train_inputs_batches) # call function 
                # mask = decoder_labels != padding_index
                # num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                # loss = self.loss_function(probs, decoder_labels, mask)
                # probs_binary = tf.cast(tf.greater_equal(probs, 0.5), tf.int64) # case for binary accuracy calculation
                # probs_binary = tf.cast(tf.greater_equal(probs, tf.cast(0.5, probs.dtype)), tf.int64)
                # print("these are the train labels batches: ", train_labels_batches)
                # print("train label batches shape is: ", train_labels_batches.shape)
                # print("these are the probs: ", probs)
                # print("probs shape is: ", tf.shape(probs))
                # print("reshaped probs: ", tf.squeeze(probs))
                # print("reshaped probs: ", tf.math.reduce_mean(tf.squeeze(probs), axis=1))
                sus_probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1) # reshaping things
                # loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs)
                predictions = tf.cast(sus_probs >= 0.5, tf.float32)
                true_labels = tf.cast(train_labels_batches, tf.float32)
                loss = tf.keras.losses.binary_crossentropy(true_labels, sus_probs)
                losses.append(loss)
                accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
                # accuracy = tf.keras.metrics.binary_accuracy(train_labels_batches, sus_probs)
                accuracies.append(accuracy)
                # loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs_binary)
                # print("average loss is: ", tf.math.reduce_mean(loss))
                # losses.append(tf.math.reduce_mean(loss))
                # accuracy = tf.keras.metrics.binary_accuracy(train_labels_batches, probs_binary)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            # accuracy = tf.keras.metrics.binary_accuracy(labels, probs, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            # total_seen += 1
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

        # avg_loss = float(total_loss / total_seen)
        # avg_loss = total_loss / total_seen
        # avg_acc = float(total_correct / total_seen)
        # avg_prp = np.exp(avg_loss)
        # print(f"\nTrain Epoch\t Loss: {avg_loss:.3f}\t Acc: {avg_acc:.3f}\t Perp: {avg_prp:.3f}")

        # return avg_loss, avg_acc, avg_prp
        # print("total loss is ", total_loss)
        # print("average loss is: ", tf.math.reduce_mean(losses))
        # print("average loss is: ", avg_loss)
        return tf.math.reduce_mean(losses), tf.math.reduce_mean(accuracies)
        avg_loss = total_loss / (len(reviews_shuffled) / batch_size)
        return avg_loss
        # return total_loss

    def test(self, reviews, labels, batch_size=30):
        """
        DO NOT CHANGE; Use as inspiration

        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        # num_batches = int(len(reviews) / batch_size)

        # total_loss = total_seen = total_correct = 0
        print("in coLSTM test")
        # total_loss = 0
        # for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            # start = end - batch_size
            # batch_image_features = test_image_features[start:end, :]
            # decoder_input = test_captions[start:end, :-1]
            # decoder_labels = test_captions[start:end, 1:]
            
            # start = end - batch_size
            # train_inputs_batches = reviews_shuffled[b0:b1]
            # train_labels_batches = labels_shuffled[b0:b1]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            # probs = self(batch_image_features, decoder_input)
            # mask = decoder_labels != padding_index
            # num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            # loss = self.loss_function(probs, decoder_labels, mask)
            # accuracy = self.accuracy_function(probs, decoder_labels, mask)

        probs = self(reviews, training=False)
        sus_probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1) # reshaping things
        loss = tf.keras.losses.binary_crossentropy(labels, sus_probs)
        predictions = tf.cast(sus_probs >= 0.5, tf.float32)
        true_labels = tf.cast(labels, tf.float32)
        accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
        # accuracy = tf.keras.metrics.binary_accuracy(labels, probs)

            ## Compute and report on aggregated statistics
        # total_loss += loss
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

            # avg_loss = float(total_loss / total_seen)
            # avg_acc = float(total_correct / total_seen)
            # avg_prp = np.exp(avg_loss)
            # print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        # print()        
        # return avg_prp, avg_acc
        # print("******loss for test is: ", loss)
        # return total_loss
        return loss, accuracy
    

#####################################################################
### LSTM-Only Model
class LSTM(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        print("in init")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # self.window_size = window_size
        self.embed_size = 64
        self.output_size = 1 # pos or neg
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
    
    @tf.function
    def call(self, reviews, training=True):
        # print("in call")
        review_embeddings = self.embedding(reviews) # reviews need to have dimension of self.vocab_size
        # print("review embeddings shape: ", review_embeddings.shape)
        
        # review_embeddings = tf.expand_dims(review_embeddings, axis=-1)
        # print("review embeddings shape expanded: ", review_embeddings.shape)

        # cnn_output = self.cnn(review_embeddings)
        # cnn_normalized = self.batch_norm(cnn_output, training=training)
        # dropout = self.dropout(cnn_normalized, training=training)
        # cnn_pooled = self.pooling(dropout)

        # cnn_pooled = tf.reshape(cnn_pooled, [tf.shape(cnn_pooled)[0], tf.shape(cnn_pooled)[1], -1])


        lstm_out = self.lstm(review_embeddings)
        # lstm_out = self.lstm(pool_dropout)
        # lstm_out = self.lstm(review_embeddings)
        # test = tf.expand_dims(lstm_out[0], axis=-1)
        lstm_normalized = self.batch_norm(lstm_out[0], training=training)
        lstm_dropout = self.dropout(lstm_normalized, training=training)

        # print("lstm out shape: ", lstm_out[0].shape)
        # dropout = self.dropout(lstm_out[0]) # Daniel: added dropout layer
        # print("dropout shape: ", dropout)
        # dense_out = self.dense(dropout)
        dense_out = self.dense(lstm_dropout)
        # dense_out = self.dense(lstm_out[0])
        # dense_out = self.dense(lstm_dropout)
        # print("dense out shape is: ", dense_out.shape)
        # print("output size is: ", self.output_size)
        return dense_out
    

    def train(self, reviews, labels, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_images: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """

        ## TODO: Implement similar to test below.

        ## NOTE: shuffle the training examples (perhaps using tf.random.shuffle on a
        ##       range of indices spanning # of training entries, then tf.gather) 
        ##       to make training smoother over multiple epochs.

        ## NOTE: make sure you are calculating gradients and optimizing as appropriate
        ##       (similar to batch_step from HW2)
        print("in LSTM train")
        # num_batches = int(len(reviews) / batch_size)
        shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
        reviews_shuffled = tf.gather(reviews, shuffled_indices)
        labels_shuffled = tf.gather(labels, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        losses = []
        accuracies = []
        for index, end in enumerate(range(batch_size, len(reviews_shuffled)+1, batch_size)):
            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            # print("TRAINING: ", start)
            # b0 = end - batch_size
            # train_inputs_batches = reviews_shuffled[b0:end]
            # train_labels_batches = labels_shuffled[b0:end]
            train_inputs_batches = reviews_shuffled[start:end]
            train_labels_batches = labels_shuffled[start:end]
            # batch_image_features = image_features_shuffled[start:end, :]
            # decoder_input = captions_shuffled[start:end, :-1]
            # decoder_labels = captions_shuffled[start:end, 1:]

            with tf.GradientTape() as tape:
                ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
                probs = self(train_inputs_batches) # call function 
                sus_probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1) # reshaping things
                # loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs)
                predictions = tf.cast(sus_probs >= 0.5, tf.float32)
                true_labels = tf.cast(train_labels_batches, tf.float32)
                loss = tf.keras.losses.binary_crossentropy(true_labels, sus_probs)
                losses.append(loss)
                accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
                # accuracy = tf.keras.metrics.binary_accuracy(train_labels_batches, sus_probs)
                accuracies.append(accuracy)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            # accuracy = tf.keras.metrics.binary_accuracy(labels, probs, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            # total_seen += 1
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

        return tf.math.reduce_mean(losses), tf.math.reduce_mean(accuracies)
        avg_loss = total_loss / (len(reviews_shuffled) / batch_size)
        return avg_loss
        # return total_loss

    def test(self, reviews, labels, batch_size=30):
        """
        DO NOT CHANGE; Use as inspiration

        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        # num_batches = int(len(reviews) / batch_size)

        # total_loss = total_seen = total_correct = 0
        print("in LSTM test")
        # total_loss = 0
        # for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

        probs = self(reviews, training=False)
        sus_probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1) # reshaping things
        loss = tf.keras.losses.binary_crossentropy(labels, sus_probs)
        predictions = tf.cast(sus_probs >= 0.5, tf.float32)
        true_labels = tf.cast(labels, tf.float32)
        accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)

        return loss, accuracy
    
##################################################
### CNN-Only Model
class CNN(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size=128, **kwargs):
        super().__init__(**kwargs)
        print("in init")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # self.window_size = window_size
        self.embed_size = 64
        self.output_size = 1 # pos or neg
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        # Convolution Hyperparameters  CNN MIGHT BE STRIDE 1 VALID PADDING?
        # self.kernel_size = (3,3)
        # self.kernel_size = 3
        # self.num_filters = 7
        # # self.pool_size = (2, 2)
        # self.pool_size = 2
        # self.strides = None # defaults to pool size
        # self.padding = "same"
    
        # self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)
        # self.cnn = tf.keras.layers.Conv1D(
        #     filters = self.num_filters, 
        #     kernel_size=self.kernel_size, 
        #     padding=self.padding, # strides is default 1
        # )

        # self.pooling = tf.keras.layers.MaxPooling1D(
        #     pool_size=self.pool_size,
        #     strides=self.strides,
        #     padding=self.padding,
        # )

        #2D
        self.kernel_size = (3, 3) # Now specifying height and width for Conv2D
        self.num_filters = 7 # 3 # used to be 7
        self.pool_size = (2, 2) # For 2D pooling
        self.strides = (1, 1) # Stride of 1
        self.padding = "same" # This will reduce the dimension as no padding is added

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
        # self.lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True, return_state=True)
        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
        # self.dense1 = tf.keras.layers.Dense(units=self.output_size, activation="softmax")
        # self.dense2 = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")

    @tf.function
    def call(self, reviews, training=True):
        # print("in call")
        review_embeddings = self.embedding(reviews) # reviews need to have dimension of self.vocab_size
        # print("review embeddings shape: ", review_embeddings.shape)

        # cnn_output = self.cnn(review_embeddings)
        # print("cnn_output shape: ", cnn_output.shape)
        # cnn_pooled = self.pooling(cnn_output)
        # print("cnn pooled shape: ", cnn_pooled)
    
        review_embeddings = tf.expand_dims(review_embeddings, axis=-1)
        # print("review embeddings shape expanded: ", review_embeddings.shape)

        cnn_output = self.cnn(review_embeddings)
        cnn_normalized = self.batch_norm(cnn_output, training=training)
        dropout = self.dropout(cnn_normalized, training=training)
        # cnn_pooled = self.pooling(cnn_output)
        cnn_pooled = self.pooling(dropout)

        # cnn_output = tf.squeeze(cnn_output, axis=2)
        # print("cnn_output shape: ", cnn_output.shape)

        # cnn_pooled = self.pooling(cnn_normalized)

        # cnn_pooled = self.pooling(cnn_output)
        # cnn_pooled = tf.reshape(cnn_pooled, [tf.shape(cnn_pooled)[0], -1, self.num_filters])

        cnn_pooled = tf.reshape(cnn_pooled, [tf.shape(cnn_pooled)[0], tf.shape(cnn_pooled)[1], -1])
        # print("cnn pooled shape: ", cnn_pooled)
        # pool_normalized = self.batch_norm(cnn_pooled, training=training)
        # pool_dropout = self.dropout(pool_normalized, training=training)
        # pool_dropout = tf.reshape(pool_dropout, [tf.shape(pool_dropout)[0], tf.shape(pool_dropout)[1], -1])


        # lstm_out = self.lstm(cnn_pooled)
        # lstm_out = self.lstm(pool_dropout)
        # lstm_out = self.lstm(review_embeddings)
        # test = tf.expand_dims(lstm_out[0], axis=-1)
        # lstm_normalized = self.batch_norm(test, training=training)
        # lstm_dropout = self.dropout(lstm_normalized, training=training)

        # print("lstm out shape: ", lstm_out[0].shape)
        # dropout = self.dropout(lstm_out[0]) # Daniel: added dropout layer
        # print("dropout shape: ", dropout)
        # dense_out = self.dense(dropout)
        dense_out = self.dense(cnn_pooled)
        # dense_out = self.dense(lstm_out[0])
        # dense_out1 = self.dense1(lstm_out[0])
        # dense_out = self.dense2(dense_out1)
        # dense_out = self.dense(lstm_dropout)
        # print("dense out shape is: ", dense_out.shape)
        # print("output size is: ", self.output_size)
        return dense_out
        return tf.nn.softmax(dense_out) # MIGHT NOT BE NECESSARY TO SOFTMAX BECAUSE SIGMOID ACTIVATION ALREADY RETURNS PROBS

    def train(self, reviews, labels, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_images: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """

        ## TODO: Implement similar to test below.

        ## NOTE: shuffle the training examples (perhaps using tf.random.shuffle on a
        ##       range of indices spanning # of training entries, then tf.gather) 
        ##       to make training smoother over multiple epochs.

        ## NOTE: make sure you are calculating gradients and optimizing as appropriate
        ##       (similar to batch_step from HW2)
        # num_batches = int(len(reviews) / batch_size)
        shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
        reviews_shuffled = tf.gather(reviews, shuffled_indices)
        labels_shuffled = tf.gather(labels, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        losses = []
        accuracies = []
        for index, end in enumerate(range(batch_size, len(reviews_shuffled)+1, batch_size)):
            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            # print("TRAINING: ", start)
            # b0 = end - batch_size
            # train_inputs_batches = reviews_shuffled[b0:end]
            # train_labels_batches = labels_shuffled[b0:end]
            train_inputs_batches = reviews_shuffled[start:end]
            train_labels_batches = labels_shuffled[start:end]
            # batch_image_features = image_features_shuffled[start:end, :]
            # decoder_input = captions_shuffled[start:end, :-1]
            # decoder_labels = captions_shuffled[start:end, 1:]

            with tf.GradientTape() as tape:
                ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
                probs = self(train_inputs_batches) # call function 
                # mask = decoder_labels != padding_index
                # num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                # loss = self.loss_function(probs, decoder_labels, mask)
                # probs_binary = tf.cast(tf.greater_equal(probs, 0.5), tf.int64) # case for binary accuracy calculation
                # probs_binary = tf.cast(tf.greater_equal(probs, tf.cast(0.5, probs.dtype)), tf.int64)
                # print("these are the train labels batches: ", train_labels_batches)
                # print("train label batches shape is: ", train_labels_batches.shape)
                # print("these are the probs: ", probs)
                # print("probs shape is: ", tf.shape(probs))
                # print("reshaped probs: ", tf.squeeze(probs))
                # print("reshaped probs: ", tf.math.reduce_mean(tf.squeeze(probs), axis=1))
                sus_probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1) # reshaping things
                # loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs)
                predictions = tf.cast(sus_probs >= 0.5, tf.float32)
                true_labels = tf.cast(train_labels_batches, tf.float32)
                loss = tf.keras.losses.binary_crossentropy(true_labels, sus_probs)
                losses.append(loss)
                accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
                # accuracy = tf.keras.metrics.binary_accuracy(train_labels_batches, sus_probs)
                accuracies.append(accuracy)
                # loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs_binary)
                # print("average loss is: ", tf.math.reduce_mean(loss))
                # losses.append(tf.math.reduce_mean(loss))
                # accuracy = tf.keras.metrics.binary_accuracy(train_labels_batches, probs_binary)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            # accuracy = tf.keras.metrics.binary_accuracy(labels, probs, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            # total_seen += 1
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

        # avg_loss = float(total_loss / total_seen)
        # avg_loss = total_loss / total_seen
        # avg_acc = float(total_correct / total_seen)
        # avg_prp = np.exp(avg_loss)
        # print(f"\nTrain Epoch\t Loss: {avg_loss:.3f}\t Acc: {avg_acc:.3f}\t Perp: {avg_prp:.3f}")

        # return avg_loss, avg_acc, avg_prp
        # print("total loss is ", total_loss)
        # print("average loss is: ", tf.math.reduce_mean(losses))
        # print("average loss is: ", avg_loss)
        return tf.math.reduce_mean(losses), tf.math.reduce_mean(accuracies)
        avg_loss = total_loss / (len(reviews_shuffled) / batch_size)
        return avg_loss
        # return total_loss

    def test(self, reviews, labels, batch_size=30):
        """
        DO NOT CHANGE; Use as inspiration

        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        # num_batches = int(len(reviews) / batch_size)

        # total_loss = total_seen = total_correct = 0
        # total_loss = 0
        # for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            # start = end - batch_size
            # batch_image_features = test_image_features[start:end, :]
            # decoder_input = test_captions[start:end, :-1]
            # decoder_labels = test_captions[start:end, 1:]
            
            # start = end - batch_size
            # train_inputs_batches = reviews_shuffled[b0:b1]
            # train_labels_batches = labels_shuffled[b0:b1]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            # probs = self(batch_image_features, decoder_input)
            # mask = decoder_labels != padding_index
            # num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            # loss = self.loss_function(probs, decoder_labels, mask)
            # accuracy = self.accuracy_function(probs, decoder_labels, mask)

        probs = self(reviews, training=False)
        sus_probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1) # reshaping things
        loss = tf.keras.losses.binary_crossentropy(labels, sus_probs)
        predictions = tf.cast(sus_probs >= 0.5, tf.float32)
        true_labels = tf.cast(labels, tf.float32)
        accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
        # accuracy = tf.keras.metrics.binary_accuracy(labels, probs)

            ## Compute and report on aggregated statistics
        # total_loss += loss
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

            # avg_loss = float(total_loss / total_seen)
            # avg_acc = float(total_correct / total_seen)
            # avg_prp = np.exp(avg_loss)
            # print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        # print()        
        # return avg_prp, avg_acc
        # print("******loss for test is: ", loss)
        # return total_loss
        return loss, accuracy
    
##################################################
### CNN-Transformer Model
    
class CoTransformer(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=128, transformer_model_name="bert-base-uncased", **kwargs):
        super().__init__(**kwargs)
        print("in init")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = 64
        self.output_size = 1 # pos or neg
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        # Convolution Hyperparameters 
        #2D
        self.kernel_size = (3, 3) # Now specifying height and width for Conv2D
        self.num_filters = 7 # 3 # used to be 7
        self.pool_size = (2, 2) # For 2D pooling
        self.strides = (1, 1) # Stride of 1
        self.padding = "same" # This will reduce the dimension as no padding is added

        # Transformer Hyperparameters
        self.transformer_model_name = transformer_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
        self.transformer_model = TFAutoModel.from_pretrained(self.transformer_model_name)

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
        # self.lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True, return_state=True)
        self.transformer_layer = self.transformer_model.get_layer(index=0)
        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
        # self.dense2 = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
    
    @tf.function
    def call(self, reviews, training=True):
        # print("in call")
        review_embeddings = self.embedding(reviews) # reviews need to have dimension of self.vocab_size
        # print("review embeddings shape: ", review_embeddings.shape)
        
        review_embeddings = tf.expand_dims(review_embeddings, axis=-1)
        # print("review embeddings shape expanded: ", review_embeddings.shape)

        cnn_output = self.cnn(review_embeddings)
        cnn_normalized = self.batch_norm(cnn_output, training=training)
        dropout = self.dropout(cnn_normalized, training=training)
        cnn_pooled = self.pooling(dropout)

        cnn_pooled = tf.reshape(cnn_pooled, [tf.shape(cnn_pooled)[0], tf.shape(cnn_pooled)[1], -1])
        # print("cnn pooled shape: ", cnn_pooled)

        # Tokenize the input for the Transformer
        # reviews_list_of_strings = []

        # for review_tokens in reviews:
        #     # review_strings = [str(token_id) for token_id in review_tokens]
        #     review_string = tf.strings.as_string(review_tokens)
        #     # review_string = " ".join(review_string)
        #     reviews_list_of_strings.append(review_string)

        # print(reviews_list_of_strings)
        reviews_list_of_strings = tf.strings.as_string(reviews).numpy()
        print(reviews_list_of_strings)
        transformer_inputs = self.tokenizer(reviews_list_of_strings, padding=True, truncation=True, return_tensors="tf")
        
        # Transformer layer taking CNN output as input
        transformer_output = self.transformer_model(input_ids=transformer_inputs.input_ids, attention_mask=transformer_inputs.attention_mask, encoder_hidden_states=cnn_pooled)
        
        # Get the last hidden state from the Transformer
        transformer_last_hidden_state = transformer_output.last_hidden_state

        # Dense layer
        dense_out = self.dense(transformer_last_hidden_state)

        # dropout = self.dropout(lstm_out[0]) # Daniel: added dropout layer
        # print("dropout shape: ", dropout)
        # dense_out = self.dense(dropout)
        # dense_out = self.dense(lstm_out[0])
        # dense_out1 = self.dense1(lstm_out[0])
        # dense_out = self.dense2(dense_out1)
        # dense_out = self.dense(lstm_dropout)
        # print("dense out shape is: ", dense_out.shape)
        # print("output size is: ", self.output_size)
        return dense_out
        return tf.nn.softmax(dense_out) # MIGHT NOT BE NECESSARY TO SOFTMAX BECAUSE SIGMOID ACTIVATION ALREADY RETURNS PROBS
    
    def train(self, reviews, labels, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_images: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """

        ## NOTE: make sure you are calculating gradients and optimizing as appropriate
        ##       (similar to batch_step from HW2)
        print("in coLSTM train")
        # num_batches = int(len(reviews) / batch_size)
        shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
        reviews_shuffled = tf.gather(reviews, shuffled_indices)
        labels_shuffled = tf.gather(labels, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        losses = []
        accuracies = []
        for index, end in enumerate(range(batch_size, len(reviews_shuffled)+1, batch_size)):
            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            # print("TRAINING: ", start)
            # b0 = end - batch_size
            # train_inputs_batches = reviews_shuffled[b0:end]
            # train_labels_batches = labels_shuffled[b0:end]
            train_inputs_batches = reviews_shuffled[start:end]
            train_labels_batches = labels_shuffled[start:end]
            # batch_image_features = image_features_shuffled[start:end, :]
            # decoder_input = captions_shuffled[start:end, :-1]
            # decoder_labels = captions_shuffled[start:end, 1:]

            with tf.GradientTape() as tape:
                ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
                probs = self(train_inputs_batches) # call function 
                # mask = decoder_labels != padding_index
                # num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                # loss = self.loss_function(probs, decoder_labels, mask)
                # probs_binary = tf.cast(tf.greater_equal(probs, 0.5), tf.int64) # case for binary accuracy calculation
                # probs_binary = tf.cast(tf.greater_equal(probs, tf.cast(0.5, probs.dtype)), tf.int64)
                # print("these are the train labels batches: ", train_labels_batches)
                # print("train label batches shape is: ", train_labels_batches.shape)
                # print("these are the probs: ", probs)
                # print("probs shape is: ", tf.shape(probs))
                # print("reshaped probs: ", tf.squeeze(probs))
                # print("reshaped probs: ", tf.math.reduce_mean(tf.squeeze(probs), axis=1))
                sus_probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1) # reshaping things
                # loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs)
                predictions = tf.cast(sus_probs >= 0.5, tf.float32)
                true_labels = tf.cast(train_labels_batches, tf.float32)
                loss = tf.keras.losses.binary_crossentropy(true_labels, sus_probs)
                losses.append(loss)
                accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
                # accuracy = tf.keras.metrics.binary_accuracy(train_labels_batches, sus_probs)
                accuracies.append(accuracy)
                # loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs_binary)
                # print("average loss is: ", tf.math.reduce_mean(loss))
                # losses.append(tf.math.reduce_mean(loss))
                # accuracy = tf.keras.metrics.binary_accuracy(train_labels_batches, probs_binary)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            # accuracy = tf.keras.metrics.binary_accuracy(labels, probs, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            # total_seen += 1
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

        # avg_loss = float(total_loss / total_seen)
        # avg_loss = total_loss / total_seen
        # avg_acc = float(total_correct / total_seen)
        # avg_prp = np.exp(avg_loss)
        # print(f"\nTrain Epoch\t Loss: {avg_loss:.3f}\t Acc: {avg_acc:.3f}\t Perp: {avg_prp:.3f}")

        # return avg_loss, avg_acc, avg_prp
        # print("total loss is ", total_loss)
        # print("average loss is: ", tf.math.reduce_mean(losses))
        # print("average loss is: ", avg_loss)
        return tf.math.reduce_mean(losses), tf.math.reduce_mean(accuracies)
        # return total_loss

    def test(self, reviews, labels, batch_size=30):
        """
        DO NOT CHANGE; Use as inspiration

        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        # num_batches = int(len(reviews) / batch_size)

        # total_loss = total_seen = total_correct = 0
        print("in coLSTM test")
        # total_loss = 0
        # for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            # start = end - batch_size
            # batch_image_features = test_image_features[start:end, :]
            # decoder_input = test_captions[start:end, :-1]
            # decoder_labels = test_captions[start:end, 1:]
            
            # start = end - batch_size
            # train_inputs_batches = reviews_shuffled[b0:b1]
            # train_labels_batches = labels_shuffled[b0:b1]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            # probs = self(batch_image_features, decoder_input)
            # mask = decoder_labels != padding_index
            # num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            # loss = self.loss_function(probs, decoder_labels, mask)
            # accuracy = self.accuracy_function(probs, decoder_labels, mask)

        probs = self(reviews, training=False)
        sus_probs = tf.math.reduce_mean(tf.squeeze(probs), axis=1) # reshaping things
        loss = tf.keras.losses.binary_crossentropy(labels, sus_probs)
        predictions = tf.cast(sus_probs >= 0.5, tf.float32)
        true_labels = tf.cast(labels, tf.float32)
        accuracy = tf.keras.metrics.binary_accuracy(true_labels, predictions)
        # accuracy = tf.keras.metrics.binary_accuracy(labels, probs)

            ## Compute and report on aggregated statistics
        # total_loss += loss
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

            # avg_loss = float(total_loss / total_seen)
            # avg_acc = float(total_correct / total_seen)
            # avg_prp = np.exp(avg_loss)
            # print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        # print()        
        # return avg_prp, avg_acc
        # print("******loss for test is: ", loss)
        # return total_loss
        return loss, accuracy
    