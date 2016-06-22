import sys
import numpy as np
import cPickle
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

NUM_HIDDEN1 = 200
NUM_HIDDEN2 = 200
HALF_BATCH_SIZE = 10
d = 100
GEN_NUM_HIDDEN1 = 300
GEN_NUM_HIDDEN2 = 300
GEN_NUM_HIDDEN3 = 300

ACCUMULATOR_EXPAVG = 0.1

MODEL_FILENAME = 'emb_adversarial_4_en2it.pkl'

rng = check_random_state(0)
leaky_relu_gain = np.sqrt(2/(1+0.01**2))

print >> sys.stderr, 'Building computation graph for discriminator...'

input_var = T.matrix('input')
target_var = T.matrix('targer')

l_in = lasagne.layers.InputLayer(shape=(None, d), input_var=T.tanh(input_var), name='l_in')
l_in_dr = lasagne.layers.DropoutLayer(l_in, 0.2)
l_hid1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(l_in_dr, num_units=NUM_HIDDEN1, nonlinearity=lasagne.nonlinearities.leaky_rectify, W=lasagne.init.GlorotUniform(gain=leaky_relu_gain), name='l_hid1'))
l_hid1_dr = lasagne.layers.DropoutLayer(l_hid1, 0.5)
l_hid2 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(l_hid1_dr, num_units=NUM_HIDDEN2, nonlinearity=lasagne.nonlinearities.leaky_rectify, W=lasagne.init.Orthogonal(gain=leaky_relu_gain), name='l_hid2'))
l_hid2_dr = lasagne.layers.DropoutLayer(l_hid2, 0.5)
l_preout = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(l_hid2_dr, num_units=1, nonlinearity=None, name='l_preout'))
l_out = lasagne.layers.NonlinearityLayer(l_preout, nonlinearity=lasagne.nonlinearities.sigmoid, name='l_out')

prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var).mean()
accuracy = T.eq(T.ge(prediction, 0.5), target_var).mean()

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

preout_var = lasagne.layers.get_output(l_preout)
#preout_grad_norm = T.grad(preout_var.mean(), input_var).norm(2)

print >> sys.stderr, 'Compiling discriminator...'
train_fn = theano.function([input_var, target_var], [loss, accuracy], updates=updates)
eval_fn = theano.function([input_var, target_var], [loss, accuracy])
#preout_grad_norm_fn = theano.function([input_var], preout_grad_norm)

X = np.zeros((2*HALF_BATCH_SIZE, d), dtype=theano.config.floatX)
target_mat = np.vstack([np.ones((HALF_BATCH_SIZE, 1)), np.zeros((HALF_BATCH_SIZE, 1))]).astype(theano.config.floatX) # En = 1, It = 0

print >> sys.stderr, 'Building computation graph for generator...'

gen_input_var = T.matrix('gen_input_var')
#gen_adversarial_input_var = T.matrix('gen_adversarial_input')

gen_l_in = lasagne.layers.InputLayer(shape=(None, d), input_var=gen_input_var, name='gen_l_in')
gen_l_hid1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(gen_l_in, num_units=GEN_NUM_HIDDEN1, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(gain='relu'), name='gen_l_hid1'))
gen_l_hid2 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(gen_l_hid1, num_units=GEN_NUM_HIDDEN2, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Orthogonal(gain='relu'), name='gen_l_hid2'))
gen_l_hid3 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(gen_l_hid2, num_units=GEN_NUM_HIDDEN3, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Orthogonal(gain='relu'), name='gen_l_hid3'))
gen_l_out = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(gen_l_hid3, num_units=d, nonlinearity=None, W=lasagne.init.GlorotUniform(), name='gen_l_out'))

generation = lasagne.layers.get_output(gen_l_out)
generation.name='generation'
#generation_and_adv_in = T.tanh(T.concatenate([generation, gen_adversarial_input_var], axis=0))
#generation_and_adv_in.name='generation_and_adv_in'
#discriminator_raw_prediction = lasagne.layers.get_output(l_preout, T.tanh(generation))
discriminator_prediction = lasagne.layers.get_output(l_out, T.tanh(generation), deterministic=True)
gen_loss = -T.log(1.0 - discriminator_prediction).mean()
gen_loss.name='gen_loss'

gen_params = lasagne.layers.get_all_params(gen_l_out, trainable=True)
gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=0.001)

grad_norm = T.grad(gen_loss, generation).norm(2)

print >> sys.stderr, 'Compiling generator...'
gen_fn = theano.function([gen_input_var], generation)
#gen_train_fn = theano.function([gen_input_var, gen_adversarial_input_var], gen_loss, updates=gen_updates, on_unused_input='ignore')
#gen_train_fn = theano.function([gen_input_var], gen_loss, updates=gen_updates)
gen_train_pred_grad_norm_fn = theano.function([gen_input_var], [gen_loss, generation, grad_norm], updates=gen_updates)

accumulators = np.zeros(5)

def train_batch(batch_id = 1, print_every_n = 1):
	id_it = next(we_batches_it)
	id_en = next(we_batches_en)
	X[HALF_BATCH_SIZE:] = we_it.vectors[id_it]
	X[:HALF_BATCH_SIZE] = we_en.vectors[id_en]

	# Generator
	#gen_loss_val = gen_train_fn(X[:HALF_BATCH_SIZE])
	#X_gen = gen_fn(X[:HALF_BATCH_SIZE])
	#preout_grad_norm_val = preout_grad_norm_fn(X_gen)
	gen_loss_val, X_gen, preout_grad_norm_val = gen_train_pred_grad_norm_fn(X[:HALF_BATCH_SIZE])

	skip_discriminator = (batch_id > 1) and (accumulators[0] > 0.85)
	
	# Discriminator
	X[:HALF_BATCH_SIZE] = X_gen
	loss_val, accuracy_val = train_fn(X, target_mat) if not skip_discriminator else eval_fn(X, target_mat)

	if batch_id == 1:
		accumulators[:] = np.array([accuracy_val, loss_val, gen_loss_val, float(skip_discriminator), preout_grad_norm_val])
	else:
		accumulators[:] = ACCUMULATOR_EXPAVG * np.array([accuracy_val, loss_val, gen_loss_val, float(skip_discriminator), preout_grad_norm_val]) + (1.0 - ACCUMULATOR_EXPAVG) * accumulators

	if batch_id % print_every_n == 0:
		print >> sys.stderr, 'batch: %s, accuracy %s, loss %s, generator loss: %s, skip rate: %s, grad norm: %s' % (batch_id, accumulators[0], accumulators[1], accumulators[2], accumulators[3], accumulators[4])

def save_model():
	params_vals = lasagne.layers.get_all_param_values([l_out, gen_l_out])
	cPickle.dump(params_vals, open(MODEL_FILENAME, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

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
we_en.vectors = skn_en.fit_transform(we_en.vectors).astype(theano.config.floatX)
we_batches_en = we_en.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

print >> sys.stderr, 'Ready to train.'

print >> sys.stderr, 'Training...'
for i in xrange(1000000):
    train_batch(i+1, 100)


