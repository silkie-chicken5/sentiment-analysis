import numpy as np
import tensorflow as tf

class CoLSTM(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=256, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # self.window_size = window_size
        self.output_size = 1 # pos or neg
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

        # Convolution Hyperparameters
        self.kernel_size = (3,3)
        self.num_filters = 7
        self.pool_size = (2, 2)
        self.strides = None # defaults to pool size
        self.padding = "same"
        
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size)
        self.cnn = tf.keras.layers.Conv2D(
            filters = self.num_filters, 
            kernel_size=self.kernel_size, 
            padding=self.padding,
        )
        
        self.pooling = tf.keras.layers.MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )
        
        self.lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(units=self.output_size, activation="sigmoid")
    
    @tf.function
    def call(self, reviews):
        review_embeddings = self.embedding(reviews) # reviews need to have dimension of self.vocab_size
        cnn_output = self.cnn(review_embeddings)
        cnn_pooled = self.pooling(cnn_output)
        lstm_out = self.lstm(cnn_pooled)
        dense_out = self.dense(lstm_out[0])
        return tf.nn.softmax(dense_out) # returns probabilities, not logits 

#     def compile(self, optimizer, loss, metrics):
#         '''
#         Create a facade to mimic normal keras fit routine
#         '''
#         self.optimizer = optimizer
#         self.loss_function = loss 
#         self.accuracy_function = metrics[0]

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
        num_batches = int(len(reviews) / batch_size)
        shuffled_indices = tf.random.shuffle(tf.range(reviews.shape[0]))
        reviews_shuffled = tf.gather(reviews, shuffled_indices)
        labels_shuffled = tf.gather(labels, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(reviews_shuffled)+1, batch_size)):
            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            b0 = end - batch_size
            train_inputs_batches = reviews_shuffled[b0:b1]
            train_labels_batches = labels_shuffled[b0:b1]
            # batch_image_features = image_features_shuffled[start:end, :]
            # decoder_input = captions_shuffled[start:end, :-1]
            # decoder_labels = captions_shuffled[start:end, 1:]

            with tf.GradientTape() as tape:
                ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
                probs = self(train_inputs_batches) # call function 
                # mask = decoder_labels != padding_index
                # num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                # loss = self.loss_function(probs, decoder_labels, mask)
                loss = tf.keras.losses.binary_crossentropy(train_labels_batches, probs)
                accuracy = tf.keras.metrics.binary_accuracy(train_labels_batches, probs)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            # accuracy = tf.keras.metrics.binary_accuracy(labels, probs, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

        # avg_loss = float(total_loss / total_seen)
        # avg_acc = float(total_correct / total_seen)
        # avg_prp = np.exp(avg_loss)
        # print(f"\nTrain Epoch\t Loss: {avg_loss:.3f}\t Acc: {avg_acc:.3f}\t Perp: {avg_prp:.3f}")

        # return avg_loss, avg_acc, avg_prp
        return total_loss

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
        total_loss = 0
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

        probs = self(reviews)
        loss = tf.keras.losses.binary_crossentropy(labels, probs)
        accuracy = tf.keras.metrics.binary_accuracy(labels, probs)

            ## Compute and report on aggregated statistics
        total_loss += loss
            # total_seen += num_predictions
            # total_correct += num_predictions * accuracy

            # avg_loss = float(total_loss / total_seen)
            # avg_acc = float(total_correct / total_seen)
            # avg_prp = np.exp(avg_loss)
            # print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        # print()        
        # return avg_prp, avg_acc
        return total_loss
    
#     def get_config(self):
#         return {"decoder": self.decoder} ## specific to ImageCaptionModel

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
    
#     def get_config(self):
#         base_config = super().get_config()
#         config = {
#             "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
#         }
#         return {**base_config, **config}

#     @classmethod
#     def from_config(cls, config):
#         decoder_config = config.pop("decoder")
#         decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
#         return cls(decoder, **config)

# def accuracy_function(prbs, labels, mask):
#     """
#     DO NOT CHANGE

#     Computes the batch accuracy

#     :param prbs:  float tensor, word prediction probabilities [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
#     :param labels:  integer tensor, word prediction labels [BATCH_SIZE x WINDOW_SIZE]
#     :param mask:  tensor that acts as a padding mask [BATCH_SIZE x WINDOW_SIZE]
#     :return: scalar tensor of accuracy of the batch between 0 and 1
#     """
#     correct_classes = tf.argmax(prbs, axis=-1) == labels
#     accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
#     return accuracy


# def loss_function(prbs, labels, mask):
#     """
#     DO NOT CHANGE

#     Calculates the model cross-entropy loss after one forward pass
#     Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

#     :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
#     :param labels:  integer tensor, word prediction labels [batch_size x window_size]
#     :param mask:  tensor that acts as a padding mask [batch_size x window_size]
#     :return: the loss of the model as a tensor
#     """
#     masked_labs = tf.boolean_mask(labels, mask)
#     masked_prbs = tf.boolean_mask(prbs, mask)
#     scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
#     loss = tf.reduce_sum(scce)
#     return loss