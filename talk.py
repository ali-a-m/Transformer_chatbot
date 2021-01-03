import tensorflow as tf
import pickle as pkl

from text_preprocessor import preprocess_line
from Transformer_model import transformer

config = pkl.load(open(config_path, 'rb'))
print(config)

MAX_LENGTH = config['maxlen']

tokenizer = pkl.load(open(tokenizer_path, 'rb'))

loaded_model = transformer(
		vocab_size=config['vocab_size'],
		num_layers=config['num_layers'],
		units=config['units'],
		d_model=config['d_model'],
		num_heads=config['num_heads'],
		dropout=config['dropout']
	)

optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def loss_function(real, pred):
	real = tf.reshape(real, shape=(-1, MAX_LENGTH-1))

	loss = loss_obj(real, pred)

	mask = tf.cast(tf.not_equal(real, 0), tf.float32)
	loss = tf.multiply(loss, mask)

	return tf.reduce_mean(loss)

loaded_model.compile(optimizer=optimizer, loss=loss_function)
loaded_model.load_weights(model_path)

def evaluate(sentence, model):
	sentence = preprocess_line(sentence)

	seq_sent = [tokenizer.word_index[word] for word in sentence.split(' ')]
	sentence = tf.expand_dims(seq_sent, axis=0)

	output = tf.expand_dims([tokenizer.word_index['<sos>']], 0)

	for i in range(MAX_LENGTH):
		predictions = model(inputs=[sentence, output], training=False)

		# select the last word from the seq_len dimension
		predictions = predictions[:, -1:, :]
		predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

		# return the result if the predicted_id is equal to the end token
		if predicted_id == tokenizer.word_index['<eos>']:
			break

		# concatenated the predicted_id to the output which is given to the decoder
		# as its input.
		output = tf.concat([output, predicted_id], axis=-1)

	return tf.squeeze(output, axis=0)

def create_text(pred):
	text = ''

	for i in pred:
		i = i.numpy()
		if i != 0 and i != tokenizer.word_index['<sos>']:
			text += tokenizer.index_word[i] + ' '

	return text


def predict(sentence, model):
	prediction = evaluate(sentence,model)

	predicted_sentence = create_text(prediction)

	# print('Input: {}'.format(sentence))
	# print('Output: {}'.format(predicted_sentence))

	return predicted_sentence


while True:
	input_ = str(input('You: '))

	print(f'Bot: {predict(input_, loaded_model)}')