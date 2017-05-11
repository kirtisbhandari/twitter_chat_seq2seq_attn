# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")



def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

"""
Get whole file vocab, both q and a, will pass train data file
"""
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

"""
Use the created vocab and convert to token ids
will pass train_data, dev_data and test_data
q and a in the same file
"""
def data_to_token_ids(data_path, q_path, a_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not(gfile.Exists(q_path) and gfile.Exists(a_path)):
    print("Tokenizing data in %s" % data_path)
    print("Saving q_id in %s" % q_path)
    print("Saving a_id in %s" % a_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    #with open(data_path, mode="rb") as data_file:
    with gfile.GFile(data_path, mode="rb") as data_file:
      #with open(q_path, mode="w") as q_tokens_file:
      with gfile.GFile(q_path, mode="w") as q_tokens_file:
        #with open(a_path, mode="w") as a_tokens_file:
        with gfile.GFile(a_path, mode="w") as a_tokens_file:
          #while True:
          counter = 0
          lines = data_file.readlines()
          while counter +1 < len(lines):
            q_line = lines[counter]
            a_line = lines[counter+1] 
            counter += 2
            if counter % 100000 == 0:
              print("  tokenizing line %d" % counter)
            q_token_ids = sentence_to_token_ids(tf.compat.as_bytes(q_line), vocab,
                                            tokenizer, normalize_digits)
            q_tokens_file.write(" ".join([str(tok) for tok in q_token_ids]) + "\n")

            a_token_ids = sentence_to_token_ids(tf.compat.as_bytes(a_line), vocab,
                                            tokenizer, normalize_digits)
            a_tokens_file.write(" ".join([str(tok) for tok in a_token_ids]) + "\n")


def prepare_wmt_data(data_dir, vocabulary_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  print("preparing wmt data")
  # Get wmt data to the specified directory.
  train_path = os.path.join(data_dir , "train_data.txt")
  dev_path = os.path.join(data_dir , "dev_data.txt")
  test_path = os.path.join(data_dir, "test_data.txt")
  print("from utils")
  print(train_path, dev_path, test_path)

  return prepare_data(data_dir, train_path, dev_path, test_path, vocabulary_size, tokenizer)


def prepare_data(data_dir, train_path, dev_path, test_path,  vocabulary_size,
                  tokenizer=None):
  """Preapre all necessary files that are required for the training.

    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
        with gfile.GFile(a_path, mode="w") as a_tokens_file:
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
  # Create vocabularies of the appropriate sizes.
  vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
  create_vocabulary(vocab_path, train_path , vocabulary_size, tokenizer)

  # Create token ids for the training data.
  q_train_ids_path = os.path.join(data_dir, 'q_train' + (".ids%d" % vocabulary_size))
  a_train_ids_path = os.path.join(data_dir, 'a_train' + (".ids%d" % vocabulary_size))
  
  q_dev_ids_path = os.path.join(data_dir,'q_dev'  + (".ids%d" % vocabulary_size))
  a_dev_ids_path = os.path.join(data_dir,'a_dev' + (".ids%d" % vocabulary_size))

  q_test_ids_path = os.path.join(data_dir,'q_test'  + (".ids%d" % vocabulary_size))
  a_test_ids_path = os.path.join(data_dir,'a_test'  + (".ids%d" % vocabulary_size))

  data_to_token_ids(train_path, q_train_ids_path, a_train_ids_path, vocab_path, tokenizer)
  data_to_token_ids(dev_path, q_dev_ids_path, a_dev_ids_path, vocab_path, tokenizer)
  data_to_token_ids(test_path, q_test_ids_path, a_test_ids_path, vocab_path, tokenizer)



  return (q_train_ids_path, a_train_ids_path,
          q_dev_ids_path, a_dev_ids_path,
          q_test_ids_path, a_test_ids_path,
          vocab_path)


