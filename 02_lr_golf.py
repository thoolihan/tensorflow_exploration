import numpy as np
import tensorflow as tf

n = 2500
iterations = 5000
display_iteration = 1000

x_data = []
y_data = []
for i in xrange(n):
    # putts
    x1 = np.random.normal(36.0, 3.0)
    # score
    y1 = (x1 - 36) * 0.5 + 72 + np.random.normal(0.0, 0.5)
    x_data.append(x1)
    y_data.append(y1)
data = zip(x_data, y_data)

with tf.name_scope('theta'):
    W = tf.Variable(0.1, name = 'weights')
    tf.scalar_summary('weights', W)
    par = tf.Variable(90.0, name = 'par')
    tf.scalar_summary('par', par)
    putt_par = tf.Variable(30.0, name = 'par_putts')
    tf.scalar_summary('putt_par', putt_par)
    y = tf.add(tf.mul(tf.sub(x_data, putt_par), W), par)

with tf.name_scope('optimization'):
    loss = tf.reduce_mean(tf.square(y - y_data), name = 'loss')
    loss_summary = tf.scalar_summary('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

_W, _par, _putt_par, _loss = 0, 0, 0, 0

with tf.Session() as sess:

    train_writer = tf.train.SummaryWriter('logs/train', sess.graph)
    merged = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    sess.run(init)

    for step in xrange(iterations):

        _W, _par, _putt_par, _loss = sess.run([W, par, putt_par, loss])
        _, summary = sess.run([train, merged])
        if step % display_iteration == 0:
            train_writer.add_summary(summary, step)
            train_writer.flush()
            print("step: %04d\tW: %s\tpar: %s\tputt_par: %s\tcost: %s" % \
                  (step, _W, _par, _putt_par, _loss))

print "=" * 10, "done optimizing", "=" * 10

def predict(putts):
    return((float(putts) - _putt_par) * _W + _par)

def show_some_data(n, _data = data):
    scores = sorted(_data[0:n], key = lambda round: round[0])
    for putts, score in scores:
        print("putts: %d\tscore: %d\tpredicted: %d" % \
              (putts, round(score), round(predict(putts))))

print "=" * 10, "sample data", "=" * 10
show_some_data(10)
print("=" * 31)
