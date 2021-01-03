import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re
import pickle as pkl
import string
import random

def preprocess_line(line, with_tokens=True):
	line = line.lower()
	line = re.sub(r'[^a-zA-Z?.<>!\']', ' ', line)
	line = re.sub(r'[ ]+', ' ', line)

	if with_tokens:
		line = '<sos> ' + line + ' <eos>'

	return line

def read_path(path, mode, encoding=None, return_tuple=False):
	with open(path, mode, encoding=encoding) as f:
		content = f.read()
	if return_tuple:
		return tuple(line for line in content.split('\n'))

	else:
		return content

def split_conversations(data):
	conv = []
	lines = data.split('\n')

	for line in lines:
		parts = line.split('+++$+++')
		info = parts[-1].replace(' ', '').replace("'", '').strip('[]').split(',')
		conv.append(info)

	print('size of conversations list: {:,}'.format(len(conv)))
	return conv

def split_lines(data):
	lns = {}
	lines = data.split('\n')

	for line in lines:
		parts = line.split('+++$+++')
		lns[parts[0].replace(' ', '')] = preprocess_line(parts[-1])

	print('size of lines list: {:,}'.format(len(lns)))
	return lns

def get_question_answer(convs_list, lines_dict, MAX_LENGTH):
	question = []
	answer = []

	for conv in convs_list:

		if len(conv) > 1:
			# q, a = random.choice(conv[:-1]), conv[-1]
			for i in range(len(conv)-1):

				q = conv[i]
				a = conv[i+1]

				if len(lines_dict[q]) <= MAX_LENGTH and len(lines_dict[a]) <= MAX_LENGTH:
					question.append(preprocess_line(lines_dict[q], with_tokens=False))
					answer.append(preprocess_line(lines_dict[a], with_tokens=False))

	print('question size: {:,} and answer size: {:,}'.format(len(question), len(answer)))
	
	return question, answer

def write_in_file(file_name, question, answer):

	with open(f'{file_name}.txt', 'w+') as f:
		for ques, ans in zip(question, answer):
			f.write(ques+' --> '+ans+'\n')


def tokenize_text(ques_list, ans_list, oov_token, max_vocab_len):
	tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words=max_vocab_len, oov_token=oov_token)

	tokenizer.fit_on_texts(ques_list + ans_list)

	ques_seq = tokenizer.texts_to_sequences(ques_list)
	ans_seq = tokenizer.texts_to_sequences(ans_list)
	
	maxlen = max(max(map(len, ques_list)), max(map(len, ans_list)))

	ques_tensor = tf.keras.preprocessing.sequence.pad_sequences(ques_seq, maxlen=maxlen, padding='post')
	ans_tensor = tf.keras.preprocessing.sequence.pad_sequences(ans_seq, maxlen=maxlen, padding='post')

	return tokenizer, maxlen, ques_tensor, ans_tensor

def text_preprocessing(lines_path, conversations_path, vocab_maxlen, sent_maxlen, oov_token='<unk>', lines_data_enc='cp1252'):
	lines = read_path(lines_path, 'r', encoding=lines_data_enc)
	convs = read_path(conversations_path, 'r')

	lines_dict = split_lines(lines)
	convs_list = split_conversations(convs)

	questions_text, answers_text = get_question_answer(convs_list, lines_dict, sent_maxlen)

	# write_in_file(r'__data__/ex/Q&A', questions_text, answers_text)

	return tokenize_text(questions_text,
							answers_text,
							'<unk>',
							vocab_maxlen)