import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from random import shuffle

grades = 6
num_students = 60 * grades
num_staff = 3 * grades + 10

vector_set = []

for i in range(num_students):
    age = 5 + np.random.random() * grades
    vector_set.append([age,
                       age * 6.5 + np.random.randn() * (age-2)])

for i in range(num_staff):
    vector_set.append([21 + np.random.random() * 44,
                       160 + np.random.randn() * 20])

shuffle(vector_set)

vectors = tf.constant(vector_set)
k = grades + 1
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k, -1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

diffs = tf.sub(expanded_centroids, expanded_vectors)
sqr_diff = tf.square(diffs)
distances = tf.reduce_sum(sqr_diff, 2)
assignments = tf.argmin(distances, 0)



means = tf.concat(0,
    [tf.reduce_mean(
        tf.gather(
            vectors,
            tf.reshape(
                tf.where(tf.equal(assignments, c)),
                [1,-1])),
        reduction_indices = [1])
     for c in range(k)])

update_centroids = tf.assign(centroids, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

epochs = 100
for step in range(epochs):
    _, centroid_values, assignment_values = sess.run(
        [update_centroids, centroids, assignments])
    if step % 10 == 0:
        print("completed step %d of %d" % (step, epochs))


df = pd.DataFrame({"age":[v[0] for v in vector_set],
                   "weight":[v[1] for v in vector_set],
                   "cluster": assignment_values})

sns.lmplot("age", "weight", data=df, fit_reg=False, size=6, hue="cluster", legend=True)
plt.show()
