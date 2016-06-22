import sys
import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import theano
import theano.tensor as T
import lasagne

from embeddings import WordEmbeddings

#theano.config.optimizer='None'
#theano.config.exception_verbosity='high'

NUM_HIDDEN1 = 500
HALF_BATCH_SIZE = 10
d = 100

rng = check_random_state(0)

print >> sys.stderr, 'Building computation graph for discriminator...'

input_var = T.matrix('input')
target_var = T.matrix('targer')

l_in = lasagne.layers.InputLayer(shape=(None, d), input_var=input_var, name='l_in')
l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=NUM_HIDDEN1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(), name='l_hid0')
l_preout = lasagne.layers.DenseLayer(l_hid1, num_units=1, nonlinearity=None, name='l_preout')
l_out = lasagne.layers.NonlinearityLayer(l_preout, nonlinearity=lasagne.nonlinearities.sigmoid, name='l_out')

prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var).mean()
accuracy = T.eq(T.ge(prediction, 0.5), target_var).mean()

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

print >> sys.stderr, 'Compiling discriminator...'
train_fn = theano.function([input_var, target_var], [loss, accuracy], updates=updates)

X = np.zeros((2*HALF_BATCH_SIZE, d), dtype=theano.config.floatX)
target_mat = np.vstack([np.ones((HALF_BATCH_SIZE, 1)), np.zeros((HALF_BATCH_SIZE, 1))]).astype(theano.config.floatX)

print >> sys.stderr, 'Building computation graph for generator...'

gen_input_var = T.matrix('gen_input_var')
gen_adversarial_input_var = T.matrix('gen_adversarial_input')

gen_l_in = lasagne.layers.InputLayer(shape=(None, d), input_var=gen_input_var, name='gen_l_in')
gen_l_hid1 = lasagne.layers.DenseLayer(gen_l_in, num_units=d, nonlinearity=None, W=lasagne.init.GlorotUniform(), name='gen_l_hid1')
gen_l_out = gen_l_hid1

generation = lasagne.layers.get_output(gen_l_out)
generation.name='generation'
generation_and_adv_in = T.concatenate([generation, gen_adversarial_input_var], axis=0)
generation_and_adv_in.name='generation_and_adv_in'
discriminator_raw_prediction = lasagne.layers.get_output(l_preout, generation_and_adv_in)
gen_loss = discriminator_raw_prediction.mean()
gen_loss.name='gen_loss'

gen_params = lasagne.layers.get_all_params(gen_l_out, trainable=True)
gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=0.001)

print >> sys.stderr, 'Compiling generator...'
gen_fn = theano.function([gen_input_var], generation)
gen_train_fn = theano.function([gen_input_var, gen_adversarial_input_var], gen_loss, updates=gen_updates)

def train_batch(batch_id = 1, print_every_n = 1):
	id_it = next(we_batches_it)
	id_en = next(we_batches_en)
	X[:HALF_BATCH_SIZE] = we_it.vectors[id_it]
	X[HALF_BATCH_SIZE:] = we_en.vectors[id_en]

	# Generator
	gen_loss_val = gen_train_fn(X[:HALF_BATCH_SIZE], X[HALF_BATCH_SIZE:])
	X_gen = gen_fn(X[:HALF_BATCH_SIZE])
	
	# Discriminator
	X[:HALF_BATCH_SIZE] = X_gen
	loss_val, accuracy_val = train_fn(X, target_mat)
	if batch_id % print_every_n == 0:
		print >> sys.stderr, 'batch: %s, accuracy %s, loss %s, generator loss: %s' % (batch_id, accuracy_val, loss_val, gen_loss_val)

print >> sys.stderr, 'Loading Italian embeddings...'
we_it = WordEmbeddings()
we_it.load_from_word2vec('./it')
we_it.downsample_frequent_words()
skn_it = StandardScaler()
we_it.vectors = skn_it.fit_transform(we_it.vectors).astype(theano.config.floatX)
we_batches_it = we_it.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

print >> sys.stderr, 'Loading English embeddings...'
we_en = WordEmbeddings()
we_en.load_from_word2vec('./en')
we_en.downsample_frequent_words()
skn_en = StandardScaler()
we_en.vectors = skn_it.fit_transform(we_en.vectors).astype(theano.config.floatX)
we_batches_en = we_en.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

print >> sys.stderr, 'Ready to train.'

print >> sys.stderr, 'Training...'
for i in xrange(1000000):
    train_batch(i+1, 100)


