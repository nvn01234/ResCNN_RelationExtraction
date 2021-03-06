#! /usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time
import datetime
from Cnn import RECnn
from test import test
from util.DataManager import DataManager
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1486463752/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("sequence_length", 100, "Sequence length (default: 100)")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes (default: '3')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ====================
datamanager = DataManager(FLAGS.sequence_length)
testing_data = datamanager.load_testing_data()

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_p1 = graph.get_operation_by_name("input_p1").outputs[0]
        input_p2 = graph.get_operation_by_name("input_p2").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        s = graph.get_operation_by_name("output/scores").outputs[0]
        p = graph.get_operation_by_name("output/predictions").outputs[0]


        test(testing_data, input_x, input_p1, input_p2, s, p, dropout_keep_prob, datamanager, sess, -1)
