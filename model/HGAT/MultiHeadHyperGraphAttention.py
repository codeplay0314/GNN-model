from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class HyperGraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel_user = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_user",
        )
        self.kernel_video = self.add_weight(
            shape=(input_shape[1][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_video",
        )
        self.kernel_edge = self.add_weight(
            shape=(input_shape[2][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_edge",
        )
        self.kernel_user_attention = self.add_weight(
            shape=(self.units * 4, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_user_attention",
        )
        # self.kernel_svid_attention = self.add_weight(
        #     shape=(self.units * 4, 1),
        #     trainable=True,
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     name="kernel_svid_attention",
        # )
        # self.kernel_tvid_attention = self.add_weight(
        #     shape=(self.units * 4, 1),
        #     trainable=True,
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     name="kernel_tvid_attention",
        # )
        self.built = True

    def call(self, inputs):
        user_states, video_states, edge_states, hyper_edges = inputs

        # Linearly transform node states
        user_states_transformed = tf.matmul(user_states, self.kernel_user)
        video_states_transformed = tf.matmul(video_states, self.kernel_video)
        edge_states_trasformed = tf.matmul(edge_states, self.kernel_edge)

        # (1) Compute pair-wise attention scores
        hyper_edge_states = tf.concat([
            tf.gather(user_states_transformed, hyper_edges[:,0]),
            tf.gather(video_states_transformed, hyper_edges[:,1]),
            tf.gather(video_states_transformed, hyper_edges[:,2]),
            edge_states_trasformed
            ], axis=1)
        user_attention_scores = tf.nn.leaky_relu(
            tf.matmul(hyper_edge_states, self.kernel_user_attention)
        )
        user_attention_scores = tf.squeeze(user_attention_scores, -1)
        # svid_attention_scores = tf.nn.leaky_relu(
        #     tf.matmul(hyper_edge_states, self.kernel_svid_attention)
        # )
        # svid_attention_scores = tf.squeeze(svid_attention_scores, -1)
        # tvid_attention_scores = tf.nn.leaky_relu(
        #     tf.matmul(hyper_edge_states, self.kernel_tvid_attention)
        # )
        # tvid_attention_scores = tf.squeeze(tvid_attention_scores, -1)

        # (2) Normalize attention scores
        user_attention_scores = tf.math.exp(tf.clip_by_value(user_attention_scores, -2, 2))
        user_attention_scores_sum = tf.math.unsorted_segment_sum(
            data=user_attention_scores,
            segment_ids=hyper_edges[:, 0],
            num_segments=tf.reduce_max(hyper_edges[:, 0]) + 1,
        )
        user_attention_scores_sum = tf.repeat(
            user_attention_scores_sum, tf.math.bincount(tf.cast(hyper_edges[:, 0], "int32"))
        )
        user_attention_scores_norm = user_attention_scores / user_attention_scores_sum
        
        # svid_attention_scores = tf.math.exp(tf.clip_by_value(svid_attention_scores, -2, 2))
        # svid_attention_scores_sum = tf.math.unsorted_segment_sum(
        #     data=svid_attention_scores,
        #     segment_ids=hyper_edges[:, 1],
        #     num_segments=tf.reduce_max(hyper_edges[:, 1]) + 1,
        # )
        # svid_attention_scores_sum = tf.repeat(
        #     svid_attention_scores_sum, tf.math.bincount(tf.cast(hyper_edges[:, 1], "int32"))
        # )
        # svid_attention_scores_norm = svid_attention_scores / svid_attention_scores_sum
     
        # tvid_attention_scores = tf.math.exp(tf.clip_by_value(tvid_attention_scores, -2, 2))
        # tvid_attention_scores_sum = tf.math.unsorted_segment_sum(
        #     data=tvid_attention_scores,
        #     segment_ids=hyper_edges[:, 2],
        #     num_segments=tf.reduce_max(hyper_edges[:, 2]) + 1,
        # )
        # tvid_attention_scores_sum = tf.repeat(
        #     tvid_attention_scores_sum, tf.math.bincount(tf.cast(hyper_edges[:, 2], "int32"))
        # )
        # tvid_attention_scores_norm = tvid_attention_scores / tvid_attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        user_states = tf.math.unsorted_segment_sum(
            data=edge_states_trasformed * user_attention_scores_norm[:, tf.newaxis],
            segment_ids=hyper_edges[:, 0],
            num_segments=tf.shape(user_states)[0],
        )
        # svid_states = tf.math.unsorted_segment_sum(
        #     data=edge_states_trasformed * svid_attention_scores_norm[:, tf.newaxis],
        #     segment_ids=hyper_edges[:, 1],
        #     num_segments=tf.shape(video_states)[0],
        # )
        # tvid_states = tf.math.unsorted_segment_sum(
        #     data=edge_states_trasformed * tvid_attention_scores_norm[:, tf.newaxis],
        #     segment_ids=hyper_edges[:, 2],
        #     num_segments=tf.shape(video_states)[0],
        # )
        # video_states = (svid_states + tvid_states) / 2
        ## // Only User Node
        # return user_states, video_states
        return user_states

class MultiHeadHyperGraphAttention(layers.Layer):
    def __init__(self, units, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_layers = [HyperGraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        user_states, video_states, edge_states, hyper_edges = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([user_states, video_states, edge_states, hyper_edges])
            for attention_layer in self.attention_layers
        ]

        # Concatenate or average the node states from each head
        ## // Only User Node
        # user_states = []
        # video_states = []
        # for output in outputs:
        #     user_states.append(output[0])
        #     video_states.append(output[1])
        # user_states = tf.concat(user_states, axis=-1)
        # video_states = tf.concat(video_states, axis=-1)
        user_states = tf.concat(outputs, axis=-1)

        # Activate and return node states
        ## // Only User Node
        # return tf.nn.relu(user_states), tf.nn.relu(video_states)
        return tf.nn.relu(user_states)
