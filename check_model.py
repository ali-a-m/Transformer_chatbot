import os
from Transformer_model2 import *

LOG_DIR = '__logs__/model_architecture_plot'

def plot_model(Model, filename):

	tf.keras.utils.plot_model(Model, to_file=os.path.join(LOG_DIR, f'{filename}.png'), show_shapes=True)

# encoder
sample_encoder_layer = encoder_layer(
	units=512,
	d_model=128,
	num_heads=4,
	dropout=0.3,
	name="sample_encoder_layer")

sample_encoder = encoder(
	vocab_size=8192,
	num_layers=2,
	units=512,
	d_model=128,
	num_heads=4,
	dropout=0.3,
	name="sample_encoder")

plot_model(sample_encoder_layer, 'sample_encoder_layer')
plot_model(sample_encoder, 'sample_encoder')

# decoder
sample_decoder_layer = decoder_layer(
	units=512,
	d_model=128,
	num_heads=4,
	dropout=0.3,
	name="sample_decoder_layer")

sample_decoder = decoder(
	vocab_size=8192,
	num_layers=2,
	units=512,
	d_model=128,
	num_heads=4,
	dropout=0.3,
	name="sample_decoder")

plot_model(sample_decoder_layer, 'sample_decoder_layer')
plot_model(sample_decoder, 'sample_decoder')

# transformer

sample_transformer = transformer(
	vocab_size=8192,
	num_layers=4,
	units=512,
	d_model=128,
	num_heads=4,
	dropout=0.3,
	name="sample_transformer")

plot_model(sample_transformer, 'sample_transformer')