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

DISCR_HIDDEN_DIM = 500
DISCR_NUM_HIDDEN_LAYERS = 10
HALF_BATCH_SIZE = 128
d = 100
GEN_NUM_HIDDEN1 = 300
GEN_NUM_HIDDEN2 = 300
GEN_NUM_HIDDEN3 = 300

ACCUMULATOR_EXPAVG = 0.1

MODEL_FILENAME = 'emb_adversarial_5_en2it.pkl'

rng = check_random_state(0)
leaky_relu_gain = np.sqrt(2/(1+0.01**2))

class Discriminator(object):
	def __init__(self, embedding_dim=100, num_hidden_layers=2, hidden_dim=200, in_dropout_p=0.2, hidden_dropout_p=0.5, update_hyperparams={'learning_rate': 0.01}):
		self.embedding_dim = embedding_dim
		self.num_hidden_layers = num_hidden_layers
		self.hidden_dim = hidden_dim
		self.in_dropout_p = in_dropout_p
		self.hidden_dropout_p = update_hyperparams
	
		print >> sys.stderr, 'Building computation graph for discriminator...'		
		self.input_var = T.matrix('input')
		self.target_var = T.matrix('targer')

		self.l_in = lasagne.layers.InputLayer(shape=(None, self.embedding_dim), input_var=T.tanh(self.input_var), name='l_in')
		self.l_in_dr = lasagne.layers.DropoutLayer(self.l_in, 0.2)
		self.layers = [self.l_in, self.l_in_dr]
		for i in xrange(self.num_hidden_layers):
			l_hid = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(self.layers[-1], num_units=self.hidden_dim, nonlinearity=lasagne.nonlinearities.leaky_rectify, W=lasagne.init.GlorotUniform(gain=leaky_relu_gain), name=('l_hid_%s' % i)))
			l_hid_dr = lasagne.layers.DropoutLayer(l_hid, 0.5)
			self.layers.append(l_hid)
			self.layers.append(l_hid_dr)
		self.l_preout = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(self.layers[-1], num_units=1, nonlinearity=None, name='l_preout'))
		self.l_out = lasagne.layers.NonlinearityLayer(self.l_preout, nonlinearity=lasagne.nonlinearities.sigmoid, name='l_out')

		self.prediction = lasagne.layers.get_output(self.l_out)
		self.loss = lasagne.objectives.binary_crossentropy(self.prediction, self.target_var).mean()
		self.accuracy = T.eq(T.ge(self.prediction, 0.5), self.target_var).mean()

		self.params = lasagne.layers.get_all_params(self.l_out, trainable=True)
		self.updates = lasagne.updates.adam(self.loss, self.params, **update_hyperparams)

		print >> sys.stderr, 'Compiling discriminator...'
		self.train_fn = theano.function([self.input_var, self.target_var], [self.loss, self.accuracy], updates=self.updates)
		self.eval_fn = theano.function([self.input_var, self.target_var], [self.loss, self.accuracy])

discriminator_0 = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS)
discriminator_1 = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS)

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
discriminator_prediction = lasagne.layers.get_output(discriminator_0.l_out, T.tanh(generation), deterministic=True)
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

accumulators = np.zeros(7)

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
	loss_val, accuracy_val         = discriminator_0.train_fn(X, target_mat) if not skip_discriminator else discriminator_0.eval_fn(X, target_mat)
	alt_loss_val, alt_accuracy_val = discriminator_1.train_fn(X, target_mat) if not skip_discriminator else discriminator_1.eval_fn(X, target_mat)

	if batch_id == 1:
		accumulators[:] = np.array([accuracy_val, loss_val, alt_accuracy_val, alt_loss_val, gen_loss_val, float(skip_discriminator), preout_grad_norm_val])
	else:
		accumulators[:] = ACCUMULATOR_EXPAVG * np.array([accuracy_val, loss_val, alt_accuracy_val, alt_loss_val, gen_loss_val, float(skip_discriminator), preout_grad_norm_val]) + (1.0 - ACCUMULATOR_EXPAVG) * accumulators

	if batch_id % print_every_n == 0:
		print >> sys.stderr, 'batch: %s, accuracy: %s, loss: %s, alt acc: %s, alt loss: %s, generator loss: %s, skip rate: %s, grad norm: %s' % tuple([batch_id] + accumulators.tolist())

def save_model():
	params_vals = lasagne.layers.get_all_param_values([discriminator_0.l_out, discriminator_1.l_out, gen_l_out])
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
we_en.vectors = skn_it.fit_transform(we_en.vectors).astype(theano.config.floatX)
we_batches_en = we_en.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

print >> sys.stderr, 'Ready to train.'

print >> sys.stderr, 'Training...'
for i in xrange(10000000):
	train_batch(i+1, 100)
	if ((i+1) % 10000) == 0:
		print >> sys.stderr, 'Saving model...'
		save_model()
print >> sys.stderr, 'Saving model...'
save_model()



