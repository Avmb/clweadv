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

DISCR_HIDDEN_DIM = 300
DISCR_NUM_HIDDEN_LAYERS = 2
HALF_BATCH_SIZE = 128
d = 100
GEN_HIDDEN_DIM = 600
GEN_NUM_HIDDEN_LAYERS = 2
LATENT_DIM = 100
#ADV_0_PENALTY = 0.2
ADV_0_PENALTY = 0.1
ADV_1_PENALTY = 0.1
ADV_01_PENALTY = 0.9

ACCUMULATOR_EXPAVG = 0.1

MODEL_FILENAME = 'emb_multidr_adversarial_resnet_cos_autoenc_cos_2_en2it.pkl'

rng = check_random_state(0)
leaky_relu_gain = np.sqrt(2/(1+0.01**2))

def cosine_sim(a_mat, b_mat):
	dp = (a_mat * b_mat).sum(axis=1)
	a_norm = a_mat.norm(2, axis=1)
	b_norm = b_mat.norm(2, axis=1)
	return dp / (a_norm * b_norm)

class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, min_sigma=0.0, rng=None, **kwargs):
        self.rng = rng if rng else T.shared_randomstreams.RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
	self.min_sigma = min_sigma
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + (T.exp(logsigma) + self.min_sigma) * self.rng.normal(shape)

class Generator(object):
	def __init__(self, embedding_dim=100, num_hidden_layers=2, hidden_dim=200, latent_dim=10, in_dropout_p=0.2, hidden_dropout_p=0.5, min_std=1e-4, prior_std=0.1, rng=None, update_hyperparams={'learning_rate': 0.01}):
		self.embedding_dim = embedding_dim
		self.num_hidden_layers = num_hidden_layers
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.in_dropout_p = in_dropout_p
		self.hidden_dropout_p =  hidden_dropout_p
		self.min_std = min_std
		self.prior_std = prior_std
		self.rng = rng if rng else T.shared_randomstreams.RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
		self.update_hyperparameters = update_hyperparams

		print >> sys.stderr, 'Building computation graph for generator...'		
		self.input_var = T.matrix('gen_input')

		# Encoder
		self.gen_l_enc_in = lasagne.layers.InputLayer(shape=(None, self.embedding_dim), input_var=self.input_var, name='gen_l_enc_in')
		self.gen_l_enc_in_dr = lasagne.layers.DropoutLayer(self.gen_l_enc_in, self.in_dropout_p)
		self.encoder_layers = [self.gen_l_enc_in, self.gen_l_enc_in_dr]
		for i in xrange(self.num_hidden_layers):
			l_enc_hidden = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(self.encoder_layers[-1], num_units=self.num_hidden_layers, nonlinearity=lasagne.nonlinearities.leaky_rectify, W=lasagne.init.GlorotUniform(gain=leaky_relu_gain), name=('gen_l_enc_hidden_%s' % i)))
			l_enc_hidden_dr = lasagne.layers.DropoutLayer(l_enc_hidden, self.hidden_dropout_p)
			self.encoder_layers.append(l_enc_hidden)
			self.encoder_layers.append(l_enc_hidden_dr)
		self.gen_l_enc_mu = lasagne.layers.DenseLayer(self.encoder_layers[-1], num_units=self.latent_dim, nonlinearity=None, W=lasagne.init.GlorotUniform(), name=('gen_l_enc_mu'))
		#self.gen_l_enc_logstd = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(self.encoder_layers[-1], num_units=self.latent_dim, nonlinearity=None, W=lasagne.init.GlorotUniform(), name=('gen_l_enc_logstd')))
		#self.gen_l_enc_latent = GaussianSampleLayer(self.gen_l_enc_mu, self.gen_l_enc_logstd, self.min_std, self.rng)
		#self.gen_latent = lasagne.layers.get_output(self.gen_l_enc_latent)
		self.gen_latent_mu = lasagne.layers.get_output(self.gen_l_enc_mu)
		self.gen_latent = self.gen_latent_mu
		#self.gen_enc_params = lasagne.layers.get_all_params(self.gen_l_enc_latent, trainable=True)
		self.gen_enc_params = lasagne.layers.get_all_params(self.gen_l_enc_mu, trainable=True)

		# Decoder
		#self.gen_l_dec_in = lasagne.layers.InputLayer(shape=(None, self.latent_dim), input_var=self.input_var, name='gen_l_dec_in')
		#self.decoder_layers = [self.gen_l_dec_in]
		self.decoder_layers = [self.gen_l_enc_mu]
		for i in xrange(self.num_hidden_layers):
			l_dec_hidden = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(self.decoder_layers[-1], num_units=self.num_hidden_layers, nonlinearity=lasagne.nonlinearities.leaky_rectify, W=lasagne.init.GlorotUniform(gain=leaky_relu_gain), name=('gen_l_dec_hidden_%s' % i)))
			self.decoder_layers.append(l_dec_hidden)
		self.gen_l_dec_out =lasagne.layers.DenseLayer(self.decoder_layers[-1], num_units=self.embedding_dim, nonlinearity=None, W=lasagne.init.GlorotUniform(), name=('gen_l_dec_out'))
		#self.gen_decoded = lasagne.layers.get_output(self.gen_l_dec_out)
		#self.gen_dec_params = lasagne.layers.get_all_params(self.gen_l_dec_out, trainable=True)

		# Autoencoder
		#self.gen_reconstruction = lasagne.layers.get_output(self.gen_l_dec_out, self.gen_latent)
		self.gen_reconstruction = lasagne.layers.get_output(self.gen_l_dec_out)
		self.reconstruction_loss = 1.0 - cosine_sim(self.input_var, self.gen_reconstruction).mean()
		self.reconstruction_loss.name='reconstruction_loss'
		#self.gen_params = lasagne.utils.unique(self.gen_enc_params + self.gen_dec_params)
		self.gen_params = lasagne.layers.get_all_params(self.gen_l_dec_out, trainable=True)
		#self.get_param_vals = lambda : lasagne.layers.get_all_param_values([self.gen_l_enc_mu, self.gen_l_dec_out])
		self.get_param_vals = lambda : lasagne.layers.get_all_param_values([self.gen_l_dec_out])
		
		# Prior
		prior_shape = (self.input_var.shape[0], self.latent_dim)
		self.gen_prior = self.prior_std * self.rng.normal(prior_shape)

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

discriminator_0_tr = Discriminator(LATENT_DIM, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.5, hidden2out_dropout_p=0.2, update_hyperparams={'learning_rate': 0.001})
discriminator_0_ev = Discriminator(LATENT_DIM, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.5, hidden2out_dropout_p=0.2, update_hyperparams={'learning_rate': 0.001})
discriminator_1_tr = Discriminator(LATENT_DIM, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.5, hidden2out_dropout_p=0.2, update_hyperparams={'learning_rate': 0.001})
discriminator_1_ev = Discriminator(LATENT_DIM, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.5, hidden2out_dropout_p=0.2, update_hyperparams={'learning_rate': 0.001})
discriminator_01_tr = Discriminator(LATENT_DIM, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.5, hidden2out_dropout_p=0.2, update_hyperparams={'learning_rate': 0.001})
discriminator_01_ev = Discriminator(LATENT_DIM, DISCR_NUM_HIDDEN_LAYERS, DISCR_NUM_HIDDEN_LAYERS, in_dropout_p=0.0, hidden_dropout_p=0.5, hidden2out_dropout_p=0.2, update_hyperparams={'learning_rate': 0.001})

gen_0 = Generator(d, GEN_NUM_HIDDEN_LAYERS, GEN_HIDDEN_DIM, latent_dim=LATENT_DIM)
gen_1 = Generator(d, GEN_NUM_HIDDEN_LAYERS, GEN_HIDDEN_DIM, latent_dim=LATENT_DIM)

gen_0_discriminator_prediction = lasagne.layers.get_output(discriminator_0_tr.l_out, gen_0.gen_latent)
gen_1_discriminator_prediction = lasagne.layers.get_output(discriminator_1_tr.l_out, gen_1.gen_latent)
gen_01_discriminator_prediction = lasagne.layers.get_output(discriminator_01_tr.l_out, gen_0.gen_latent)
adv_gen_0_loss = -T.log(1.0 - gen_0_discriminator_prediction).mean()
adv_gen_1_loss = -T.log(1.0 - gen_1_discriminator_prediction).mean()
adv_gen_01_loss = -T.log(1.0 - gen_01_discriminator_prediction).mean()
adv_gen_0_loss.name='adv_gen_0_loss'
adv_gen_1_loss.name='adv_gen_1_loss'
adv_gen_01_loss.name='adv_gen_01_loss'

adv_0_grad_norm = T.grad(adv_gen_0_loss, gen_0.gen_latent).norm(2, axis=1).mean()
adv_1_grad_norm = T.grad(adv_gen_1_loss, gen_1.gen_latent).norm(2, axis=1).mean()
adv_01_grad_norm = T.grad(adv_gen_01_loss, gen_0.gen_latent).norm(2, axis=1).mean()

gen_0_loss = gen_0.reconstruction_loss + ADV_0_PENALTY * adv_gen_0_loss + ADV_01_PENALTY * adv_gen_01_loss
gen_1_loss = gen_1.reconstruction_loss + ADV_1_PENALTY * adv_gen_1_loss

gen_0_updates = lasagne.updates.adam(gen_0_loss, gen_0.gen_params, learning_rate=0.01)
gen_1_updates = lasagne.updates.adam(gen_1_loss, gen_1.gen_params, learning_rate=0.01)

print >> sys.stderr, 'Compiling generators...'
gen_0_eval_fn = theano.function([gen_0.input_var], [gen_0_loss, gen_0.reconstruction_loss, adv_gen_0_loss, adv_0_grad_norm, gen_0.gen_latent, gen_0.gen_prior, adv_gen_01_loss, adv_01_grad_norm])
gen_1_eval_fn = theano.function([gen_1.input_var], [gen_1_loss, gen_1.reconstruction_loss, adv_gen_1_loss, adv_1_grad_norm, gen_1.gen_latent, gen_1.gen_prior])
gen_0_train_fn = theano.function([gen_0.input_var], [gen_0_loss, gen_0.reconstruction_loss, adv_gen_0_loss, adv_0_grad_norm, gen_0.gen_latent, gen_0.gen_prior, adv_gen_01_loss, adv_01_grad_norm], updates=gen_0_updates)
gen_1_train_fn = theano.function([gen_1.input_var], [gen_1_loss, gen_1.reconstruction_loss, adv_gen_1_loss, adv_1_grad_norm, gen_1.gen_latent, gen_1.gen_prior], updates=gen_1_updates)

gen_0_X = np.zeros((HALF_BATCH_SIZE, d), dtype=theano.config.floatX)
gen_1_X = np.zeros((HALF_BATCH_SIZE, d), dtype=theano.config.floatX)
discr_0_X = np.zeros((2*HALF_BATCH_SIZE, LATENT_DIM), dtype=theano.config.floatX)
discr_1_X = np.zeros((2*HALF_BATCH_SIZE, LATENT_DIM), dtype=theano.config.floatX)
discr_01_X = np.zeros((2*HALF_BATCH_SIZE, LATENT_DIM), dtype=theano.config.floatX)
target_mat = np.vstack([np.ones((HALF_BATCH_SIZE, 1)), np.zeros((HALF_BATCH_SIZE, 1))]).astype(theano.config.floatX) # En = 1, It = 0

accumulators_gen_0 = np.zeros(5)
accumulators_gen_1 = np.zeros(5)
accumulators_discr_0 = np.zeros(3)
accumulators_discr_1 = np.zeros(3)
accumulators_discr_01 = np.zeros(5)

def train_batch(batch_id = 1, print_every_n = 1):
	id_en = next(we_batches_en)
	id_it = next(we_batches_it)
	gen_0_X[:] = we_en.vectors[id_en]
	gen_1_X[:] = we_it.vectors[id_it]

	skip_generator_0 = (batch_id > 1) and (accumulators_discr_0[0] < 0.51)
	skip_generator_1 = (batch_id > 1) and (accumulators_discr_1[0] < 0.51)

	# Generator
	gen_0_loss_val, gen_0_recon_loss_val, gen_0_adv_loss_val, gen_0_grad_norm_val, gen_0_latent_sample, gen_0_prior_sample, gen_01_adv_loss_val, gen_01_grad_norm_val = gen_0_train_fn(gen_0_X) if not skip_generator_0 else gen_0_eval_fn(gen_0_X)
	gen_1_loss_val, gen_1_recon_loss_val, gen_1_adv_loss_val, gen_1_grad_norm_val, gen_1_latent_sample, gen_1_prior_sample = gen_1_train_fn(gen_1_X) if not skip_generator_1 else gen_1_eval_fn(gen_1_X)

	skip_discriminator_0 = (batch_id > 1) and (accumulators_discr_0[0] > 0.99)
	skip_discriminator_1 = (batch_id > 1) and (accumulators_discr_1[0] > 0.99)
	skip_discriminator_01 = (batch_id > 1) and (accumulators_discr_01[0] > 0.99)
	
	# Discriminator
	discr_0_X[:HALF_BATCH_SIZE] = gen_0_latent_sample
	discr_1_X[:HALF_BATCH_SIZE] = gen_1_latent_sample
	discr_0_X[HALF_BATCH_SIZE:] = gen_0_prior_sample
	discr_1_X[HALF_BATCH_SIZE:] = gen_1_prior_sample
	discr_01_X[:HALF_BATCH_SIZE] = gen_0_latent_sample
	discr_01_X[HALF_BATCH_SIZE:] = gen_1_latent_sample
	
	discr_0_tr_loss_val, discr_0_tr_accuracy_val = discriminator_0_tr.train_fn(discr_0_X, target_mat) if not skip_discriminator_0 else discriminator_0_tr.eval_fn(discr_0_X, target_mat)
	discr_0_ev_loss_val, discr_0_ev_accuracy_val = discriminator_0_ev.train_fn(discr_0_X, target_mat) if not skip_discriminator_0 else discriminator_0_ev.eval_fn(discr_0_X, target_mat)
	discr_1_tr_loss_val, discr_1_tr_accuracy_val = discriminator_1_tr.train_fn(discr_1_X, target_mat) if not skip_discriminator_1 else discriminator_1_tr.eval_fn(discr_1_X, target_mat)
	discr_1_ev_loss_val, discr_1_ev_accuracy_val = discriminator_1_ev.train_fn(discr_1_X, target_mat) if not skip_discriminator_1 else discriminator_1_ev.eval_fn(discr_1_X, target_mat)
	discr_01_tr_loss_val, discr_01_tr_accuracy_val = discriminator_01_tr.train_fn(discr_01_X, target_mat) if not skip_discriminator_01 else discriminator_01_tr.eval_fn(discr_01_X, target_mat)
	discr_01_ev_loss_val, discr_01_ev_accuracy_val = discriminator_01_ev.train_fn(discr_01_X, target_mat) if not skip_discriminator_01 else discriminator_01_ev.eval_fn(discr_01_X, target_mat)

	accumulators_gen_0_upd = np.array([gen_0_loss_val, gen_0_recon_loss_val, gen_0_adv_loss_val, gen_0_grad_norm_val, float(skip_generator_0)])
	accumulators_gen_1_upd = np.array([gen_1_loss_val, gen_1_recon_loss_val, gen_1_adv_loss_val, gen_1_grad_norm_val, float(skip_generator_1)])
	accumulators_discr_0_upd = np.array([discr_0_tr_accuracy_val, discr_0_ev_accuracy_val, float(skip_discriminator_0)])
	accumulators_discr_1_upd = np.array([discr_1_tr_accuracy_val, discr_1_ev_accuracy_val, float(skip_discriminator_1)])
	accumulators_discr_01_upd = np.array([discr_01_tr_accuracy_val, discr_01_ev_accuracy_val, float(skip_discriminator_01), gen_01_adv_loss_val, gen_01_grad_norm_val])

	if batch_id == 1:
		accumulators_gen_0[:] = accumulators_gen_0_upd
		accumulators_gen_1[:] = accumulators_gen_1_upd
		accumulators_discr_0[:] = accumulators_discr_0_upd
		accumulators_discr_1[:] = accumulators_discr_1_upd
		accumulators_discr_01[:] = accumulators_discr_01_upd
	else:
		accumulators_gen_0[:] = ACCUMULATOR_EXPAVG * accumulators_gen_0_upd + (1.0 - ACCUMULATOR_EXPAVG) * accumulators_gen_0
		accumulators_gen_1[:] = ACCUMULATOR_EXPAVG * accumulators_gen_1_upd + (1.0 - ACCUMULATOR_EXPAVG) * accumulators_gen_1
		accumulators_discr_0[:] = ACCUMULATOR_EXPAVG * accumulators_discr_0_upd + (1.0 - ACCUMULATOR_EXPAVG) * accumulators_discr_0
		accumulators_discr_1[:] = ACCUMULATOR_EXPAVG * accumulators_discr_1_upd + (1.0 - ACCUMULATOR_EXPAVG) * accumulators_discr_1
		accumulators_discr_01[:] = ACCUMULATOR_EXPAVG * accumulators_discr_01_upd + (1.0 - ACCUMULATOR_EXPAVG) * accumulators_discr_01

	if batch_id % print_every_n == 0:
		print >> sys.stderr, 'batch: %s' % batch_id
		print >> sys.stderr, 'gen 0 ',
		print_vals_0 = tuple(accumulators_discr_0.tolist() + accumulators_gen_0.tolist())
		print >> sys.stderr, 'tr acc %s, ev acc %s, dskip %s, gen loss: %s, recon loss %s, adv loss: %s, grad: %s, gskip: %s' % print_vals_0
		print >> sys.stderr, 'gen 1 ',
		print_vals_1 = tuple(accumulators_discr_1.tolist() + accumulators_gen_1.tolist())
		print >> sys.stderr, 'tr acc %s, ev acc %s, dskip %s, gen loss: %s, recon loss %s, adv loss: %s, grad: %s, gskip: %s' % print_vals_1
		print >> sys.stderr, 'd  01 ',
		print_vals_01 = tuple(accumulators_discr_01.tolist())
		print >> sys.stderr, 'tr acc %s, ev acc %s, dskip %s, adv loss: %s, grad: %s' % print_vals_01
		print >> sys.stderr, ''

def save_model():
	params_vals = lasagne.layers.get_all_param_values([discriminator_0_tr.l_out, discriminator_0_ev.l_out, discriminator_0_tr.l_out, discriminator_0_ev.l_out, discriminator_01_tr.l_out, discriminator_01_ev.l_out])
	params_vals += gen_0.get_param_vals()
	params_vals += gen_1.get_param_vals()
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



