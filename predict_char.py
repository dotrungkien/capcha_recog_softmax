import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

d = {'C': [1,0,0,0,0,0,0,0],
    'E': [0,1,0,0,0,0,0,0],
    'F': [0,0,1,0,0,0,0,0],
    'M': [0,0,0,1,0,0,0,0],
    'N': [0,0,0,0,1,0,0,0],
    'P': [0,0,0,0,0,1,0,0],
    'T': [0,0,0,0,0,0,1,0],
    'W': [0,0,0,0,0,0,0,1],}

def predict(img_path):
    img = np.array(Image.open(img_path).convert('L')).reshape(1024,)
    for i in range(1024):
        img[i] = img[i]/255.
    # Running a new session
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % save_path)

        # Resume training
        for epoch in range(7):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Second Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval(
            {x: mnist.test.images, y: mnist.test.labels}))