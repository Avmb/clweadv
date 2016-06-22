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

DISCR_HIDDEN_DIM = 600
DISCR_NUM_HIDDEN_LAYERS = 10
HALF_BATCH_SIZE = 128
d = 100
ADV_PENALTY = 1.0
ADV_PENALTY_1 = 0.1
#ADV_PENALTY = 0.1
#COS_PENALTY = 1.0
PRIOR_STD = 0.1

ACCUMULATOR_EXPAVG = 0.1

MODEL_FILENAME = 'emb_multilin_adversarial_resnet_cos_autoenc_cos_en2it.pkl'

rng = check_random_state(0)
leaky_relu_gain = np.sqrt(2/(1+0.01**2))

def cosine_sim(a_mat, b_mat):
	dp = (a_mat * b_mat).sum(axis=1)
	a_norm = a_mat.norm(2, axis=1)
	b_norm = b_mat.norm(2, axis=1)
	return dp / (a_norm * b_norm)

class Discriminator(object):
	def __init__(self, embedding_dim=100, num_hidden_layers=2, hidden_dim=200, in_dropout_p=0.2, hidden_dropout_p=0.5, hidden2out_dropout_p=0.5, update_hyperparams={'learning_rate': 0.01}):
		self.embedding_dim = embedding_dim
		self.num_hidden_layers = num_hidden_layers
		self.hidden_dim = hidden_dim
		self.in_dropout_p = in_dropout_p
		self.hidden_dropout_p = hidden_dropout_p
		self.hidden2out_dropout_p = hidden2out_dropout_p
		self.update_hyperparameters = update_hyperparams
	
		print >> sys.stderr, 'Building computation graph for discriminator...'		
		self.input_var = T.matrix('input')
		self.target_var = T.matrix('targer')

		self.l_in = lasagne.layers.InputLayer(shape=(None, self.embedding_dim), input_var=T.tanh(self.input_var), name='l_in')
		self.l_in_dr = lasagne.layers.DropoutLayer(self.l_in, self.in_dropout_p)
		self.l_prehid = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(self.l_in_dr, num_units=self.hidden_dim, nonlinearity=lasagne.nonlinearities.leaky_rectify, W=lasagne.init.GlorotUniform(gain=leaky_relu_gain), name='l_prehid'))
		self.layers = [self.l_in, self.l_in_dr, self.l_prehid]
		for i in xrange(self.num_hidden_layers):
			l_hid_predr = lasagne.layers.DropoutLayer(self.layers[-1], self.hidden_dropout_p)
			l_hid = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(l_hid_predr, num_units=self.hidden_dim, nonlinearity=lasagne.nonlinearities.leaky_rectify, W=lasagne.init.GlorotUniform(gain=leaky_relu_gain), name=('l_hid_%s' % i)))
			l_hid_sum = lasagne.layers.ElemwiseSumLayer([self.layers[-1], l_hid])
			self.layers.append(l_hid_predr)
			self.layers.append(l_hid)
			self.layers.append(l_hid_sum)

		self.l_preout_predr = lasagne.layers.DropoutLayer(self.layers[-1], self.hidden2out_dropout_p)
		self.l_preout = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(self.l_preout_predr, num_units=1, nonlinearity=None, name='l_preout'))
		self.l_out = lasagne.layers.NonlinearityLayer(self.l_preout, nonlinearity=lasagne.nonlinearities.sigmoid, name='l_out')

		self.prediction = lasagne.layers.get_output(self.l_out)
		self.loss = lasagne.objectives.binary_crossentropy(self.prediction, self.target_var).mean()
		self.accuracy = T.eq(T.ge(self.prediction, 0.5), self.target_var).mean()

		self.params = lasagne.layers.get_all_params(self.l_out, trainable=True)
		self.updates = lasagne.updates.adam(self.loss, self.params, **update_hyperparams)

		print >> sys.stderr, 'Compiling discriminator...'
		self.train_fn = theano.function([self.input_var, self.target_var], [self.loss, self.accuracy], updates=self.updates)
		self.eval_fn = theano.function([self.input_var, self.target_var], [self.loss, self.accuracy])

#discriminator_0 = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.0, hidden2out_dropout_p=0.0, update_hyperparams={'learning_rate': 0.01})
#discriminator_1 = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.0, hidden2out_dropout_p=0.0, update_hyperparams={'learning_rate': 0.01})

discriminator_0 = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.3, hidden2out_dropout_p=0.1, update_hyperparams={'learning_rate': 0.01})
discriminator_1 = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.3, hidden2out_dropout_p=0.1, update_hyperparams={'learning_rate': 0.01})

discriminator_01 = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.3, hidden2out_dropout_p=0.1, update_hyperparams={'learning_rate': 0.01})
discriminator_11 = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.3, hidden2out_dropout_p=0.1, update_hyperparams={'learning_rate': 0.01})

X = np.zeros((2*HALF_BATCH_SIZE, d), dtype=theano.config.floatX)
target_mat = np.vstack([np.ones((HALF_BATCH_SIZE, 1)), np.zeros((HALF_BATCH_SIZE, 1))]).astype(theano.config.floatX) # En = 1, It = 0
neg_target_mat = 1.0 - target_mat

print >> sys.stderr, 'Building computation graph for generator...'

gen_input_var = T.matrix('gen_input_var')
#gen_adversarial_input_var = T.matrix('gen_adversarial_input')

gen_l_in = lasagne.layers.InputLayer(shape=(None, d), input_var=gen_input_var, name='gen_l_in')
gen_l_out = lasagne.layers.DenseLayer(gen_l_in, num_units=d, nonlinearity=None, W=lasagne.init.Orthogonal(), name='gen_l_out')

generation = lasagne.layers.get_output(gen_l_out)
generation.name='generation'
deterministic_generation = lasagne.layers.get_output(gen_l_out, deterministic=True)
deterministic_generation.name='deterministic_generation'

discriminator_prediction = lasagne.layers.get_output(discriminator_0.l_out, T.tanh(generation), deterministic=True)
adv_gen_loss = -T.log(1.0 - discriminator_prediction).mean()
adv_gen_loss.name='adv_gen_loss'

#cos_gen_loss = 1.0 - cosine_sim(gen_adversarial_input_var, generation).mean()
#cos_gen_loss.name = 'cos_gen_loss'

dec_l_out = lasagne.layers.DenseLayer(gen_l_out, num_units=d, nonlinearity=None, W=gen_l_out.W.T, name='dec_l_out')

reconstruction = lasagne.layers.get_output(dec_l_out)
deterministic_reconstruction = lasagne.layers.get_output(dec_l_out, deterministic=True)
#recon_gen_loss = (gen_input_var - reconstruction).norm(2, axis=1).mean()
recon_gen_loss = 1.0 - cosine_sim(gen_input_var, reconstruction).mean()
recon_gen_loss.name='recon_gen_loss'

gen_loss = recon_gen_loss + ADV_PENALTY * adv_gen_loss #+ COS_PENALTY * cos_gen_loss
gen_loss.name='gen_loss'

gen_params = lasagne.layers.get_all_params(dec_l_out, trainable=True)
gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=0.001)
recon_gen_updates = lasagne.updates.adam(recon_gen_loss, gen_params, learning_rate=0.001)

grad_norm = T.grad(adv_gen_loss, generation).norm(2, axis=1).mean()

print >> sys.stderr, 'Compiling generator...'
gen_fn = theano.function([gen_input_var], deterministic_generation)
recon_fn = theano.function([gen_input_var], deterministic_reconstruction)
gen_train_pred_grad_norm_fn = theano.function([gen_input_var], [gen_loss, recon_gen_loss, adv_gen_loss, deterministic_generation, grad_norm], updates=gen_updates)
gen_train_recon_only_pred_grad_norm_fn = theano.function([gen_input_var], [gen_loss, recon_gen_loss, adv_gen_loss, deterministic_generation, grad_norm], updates=recon_gen_updates)
gen_eval_pred_grad_norm_fn  = theano.function([gen_input_var], [gen_loss, recon_gen_loss, adv_gen_loss, deterministic_generation, grad_norm])

print >> sys.stderr, 'Building computation graph for alt generator...'
gen_1_input_var = T.matrix('gen_1_input_var')

gen_1_l_in = lasagne.layers.InputLayer(shape=(None, d), input_var=gen_1_input_var, name='gen_1_l_in')
gen_1_l_out = lasagne.layers.DenseLayer(gen_1_l_in, num_units=d, nonlinearity=None, W=lasagne.init.Orthogonal(), name='gen_1_l_out')

generation_1 = lasagne.layers.get_output(gen_1_l_out)
generation_1.name='generation'
deterministic_generation_1 = lasagne.layers.get_output(gen_1_l_out, deterministic=True)
deterministic_generation_1.name='deterministic_generation_1'

dec_1_l_out = lasagne.layers.DenseLayer(gen_1_l_out, num_units=d, nonlinearity=None, W=gen_1_l_out.W.T, name='dec_1_l_out')

reconstruction_1 = lasagne.layers.get_output(dec_1_l_out)
deterministic_reconstruction_1 = lasagne.layers.get_output(dec_1_l_out, deterministic=True)
recon_gen_loss_1 = 1.0 - cosine_sim(gen_1_input_var, reconstruction_1).mean()
recon_gen_loss_1.name='recon_gen_loss_1'

discriminator_prediction_1 = lasagne.layers.get_output(discriminator_01.l_out, T.tanh(generation_1), deterministic=True)
adv_gen_loss_1 = -T.log(1.0 - discriminator_prediction_1).mean()
adv_gen_loss_1.name='adv_gen_loss_1'

gen_loss_1 = recon_gen_loss_1 + ADV_PENALTY_1 * adv_gen_loss_1
gen_loss_1.name='gen_loss_1'

gen_params_1 = lasagne.layers.get_all_params(dec_1_l_out, trainable=True)
gen_updates_1 = lasagne.updates.adam(gen_loss_1, gen_params_1, learning_rate=0.001)
recon_gen_updates_1 = lasagne.updates.adam(recon_gen_loss_1, gen_params_1, learning_rate=0.001)

grad_norm_1 = T.grad(adv_gen_loss_1, generation_1).norm(2, axis=1).mean()

print >> sys.stderr, 'Compiling alt generator...'
gen_1_fn = theano.function([gen_1_input_var], deterministic_generation_1)
recon_1_fn = theano.function([gen_1_input_var], deterministic_reconstruction_1)
gen_1_train_pred_grad_norm_fn = theano.function([gen_1_input_var], [gen_loss_1, recon_gen_loss_1, adv_gen_loss_1, deterministic_generation_1, grad_norm_1], updates=gen_updates_1)
gen_1_train_recon_only_pred_grad_norm_fn = theano.function([gen_1_input_var], [gen_loss_1, recon_gen_loss_1, adv_gen_loss_1, deterministic_generation_1, grad_norm_1], updates=recon_gen_updates_1)
gen_1_eval_pred_grad_norm_fn  = theano.function([gen_1_input_var], [gen_loss_1, recon_gen_loss_1, adv_gen_loss_1, deterministic_generation_1, grad_norm_1])

accumulators = np.zeros(18)

def train_batch(batch_id = 1, print_every_n = 1):
	id_it = next(we_batches_it)
	id_en = next(we_batches_en)
	X[HALF_BATCH_SIZE:] = we_it.vectors[id_it]
	X[:HALF_BATCH_SIZE] = we_en.vectors[id_en]

	skip_generator = (batch_id > 1) and (accumulators[0] < 0.51)

	# Generator
	gen_loss_val, recon_gen_loss_val, adv_gen_loss_val, X_gen, preout_grad_norm_val = gen_train_pred_grad_norm_fn(X[:HALF_BATCH_SIZE]) if not skip_generator  else gen_train_recon_only_pred_grad_norm_fn(X[:HALF_BATCH_SIZE])

	gen_loss_val_1, recon_gen_loss_1, adv_gen_loss_val_1, X_gen_1, preout_grad_norm_val_1 = gen_1_train_pred_grad_norm_fn(X[HALF_BATCH_SIZE:])

	skip_discriminator = (batch_id > 1) and (accumulators[0] > 0.99)
	
	# Discriminator
	X[:HALF_BATCH_SIZE] = X_gen
	X[HALF_BATCH_SIZE:] = X_gen_1
	loss_val, accuracy_val         = discriminator_0.train_fn(X, target_mat) if not skip_discriminator else discriminator_0.eval_fn(X, target_mat)
	alt_loss_val, alt_accuracy_val = discriminator_1.train_fn(X, target_mat) if not skip_discriminator else discriminator_1.eval_fn(X, target_mat)

	X[:HALF_BATCH_SIZE] = np.random.normal(scale=PRIOR_STD, size=(HALF_BATCH_SIZE, d))
	loss_val_1, accuracy_val_1         = discriminator_01.train_fn(X, neg_target_mat)
	alt_loss_val_1, alt_accuracy_val_1 = discriminator_11.train_fn(X, neg_target_mat)

	if batch_id == 1:
		accumulators[:] = np.array([accuracy_val, loss_val, alt_accuracy_val, alt_loss_val, gen_loss_val, recon_gen_loss_val, adv_gen_loss_val, float(skip_generator), float(skip_discriminator), preout_grad_norm_val, gen_loss_val_1, recon_gen_loss_1, adv_gen_loss_val_1, preout_grad_norm_val_1, loss_val_1, accuracy_val_1, alt_loss_val_1, alt_accuracy_val_1])
	else:
		accumulators[:] = ACCUMULATOR_EXPAVG * np.array([accuracy_val, loss_val, alt_accuracy_val, alt_loss_val, gen_loss_val, recon_gen_loss_val, adv_gen_loss_val, float(skip_generator), float(skip_discriminator), preout_grad_norm_val, gen_loss_val_1, recon_gen_loss_1, adv_gen_loss_val_1, preout_grad_norm_val_1, loss_val_1, accuracy_val_1, alt_loss_val_1, alt_accuracy_val_1]) + (1.0 - ACCUMULATOR_EXPAVG) * accumulators

	if batch_id % print_every_n == 0:
		print >> sys.stderr, 'batch: %s' % batch_id
		print >> sys.stderr, 'acc: %s, loss: %s, alt acc: %s, alt loss: %s, gloss: %s, grloss: %s, galoss: %s, gskip: %s, dskip: %s, gn: %s, agloss: %s, agrloss: %s, agaloss: %s, agn: %s, adloss: %s, adacc: %s, alt adloss: %s, alt adacc: %s' % tuple(accumulators.tolist())
		print >> sys.stderr, ''

def save_model():
	params_vals = lasagne.layers.get_all_param_values([discriminator_0.l_out, discriminator_1.l_out, gen_l_out, gen_1_l_out, discriminator_0.l_out, discriminator_1.l_out])
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
for i in xrange(10000000):
	train_batch(i+1, 100)
	if ((i+1) % 10000) == 0:
		print >> sys.stderr, 'Saving model...'
		save_model()
print >> sys.stderr, 'Saving model...'
save_model()



