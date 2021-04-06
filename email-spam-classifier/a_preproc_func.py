import os

import numpy as np

import re
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


class GetTextDataset:
	def __init__(self):
		self.files_directory = ""
		self.features = []
		self.labels = []
		self.tokens = []
		self.stemmed_feature = []
		self.lemma_feature = []
	
	def read_text_files(self, files_directory):
		'''
		Function to read multiple text files
		Return as a np.array object containing label and feature
		'''
		features = []
		labels = []

		for fd_name in sorted(os.listdir(files_directory)):
			# Save folder path name as label
			fd_path = os.path.join(files_directory, fd_name)
			if os.path.isdir(fd_path):
				for fl in sorted(os.listdir(fd_path)):
					with open(os.path.join(fd_path, fl), mode="r", encoding="utf-8", errors="ignore") as fl_read:
						features.append(fl_read.read())
						labels.append(fd_path)

		self.features = np.array(features)
		self.labels = np.array(labels)
		self.files_directory = files_directory

		return(self)
	

	def tokenize(self, stopwords):
		'''
		Function to tokenize the words, remove stopwords
		'''
		tokenized_text = []

		for corpus in self.features:
			words = word_tokenize(corpus)
			stop_words = set(stopwords)
			final_tokens = []
			for each in words:
				if each not in stop_words:
					final_tokens.append(each)
			tokenized_text.append(final_tokens)
		
		self.tokens = tokenized_text

		return(self)

	def stem_sentence(self, stemmer):
		'''
		Function to do sentence stemming
		Use tokenized sentences or else stemming will return the entire sentence as is
		'''
		stemmed_feature = []

		for token_words in self.tokens:
			stem_sentence=[]
			for word in token_words:
				stem_sentence.append(stemmer.stem(word.lower()))
				# Add whitespace to form tokens back to sentences
				stem_sentence.append(" ")
			stemmed_feature.append("".join(stem_sentence))

		self.stemmed_feature = np.array(stemmed_feature)

		return(self)
		


def main(fd="./enron1", max_feats=100):
	# set arbitrary number of maximum features that are allowed

	files_directory = fd
	emails = GetTextDataset()
	email_dataset = emails.read_text_files(files_directory)
	tokenized_dataset = email_dataset.tokenize(stopwords.words('english'))
	stemmed_dataset = tokenized_dataset.stem_sentence(SnowballStemmer("english"))

	# Create bag of words vector using sklearn.feature_extraction.text CountVectorizer
	count_vect = CountVectorizer(max_features=max_feats)
	X = count_vect.fit_transform(stemmed_dataset.stemmed_feature)

	le = LabelEncoder()
	le.fit(stemmed_dataset.labels)
	y = le.transform(stemmed_dataset.labels)

	return(X.toarray(),y)


if __name__ == '__main__':
	main()