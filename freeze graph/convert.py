import tensorflow as tf

meta_path = 'C:/Users/rashi/Desktop/Elevator/multi-digit-leap-motion/checkpoints/mlt_leap_v1-6000.meta' # Your .meta file
path = 'C:/Users/rashi/Desktop/Elevator/multi-digit-leap-motion/checkpoints/mlt_leap_v1-6000'
#output_node_names = ['output:0']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver =  tf.compat.v1.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,path)

    output_node_names = [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    # Freeze the graph
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names)

    # Save the frozen graph
    with open('mlt_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())