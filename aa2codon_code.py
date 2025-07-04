#@title Tarvittavat kirjastot ja asennukset


!git clone https://github.com/pyypyyy/aa2codon.git
!pip install --upgrade tensorflow
!pip install biopython
!pip install tensorflow-text
!pip install git+https://github.com/Benjamin-Lee/CodonAdaptationIndex.git

#!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
#!pip uninstall -y -q tensorflow keras tensorflow-estimator tensorflow-text
#!pip install -q tensorflow_datasets

import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_text

from random import randrange
import statistics
from Bio.Seq import Seq
from Bio.SeqUtils import CodonAdaptationIndex
from Bio import SeqIO

import pandas as pd
import plotly.express as px


#@title Click the arrow and provide a sequence such as "atgtttccc" to the prompt. Max lenght is 150 codons --> 450 bp { display-mode: "form" }



seq = input("Provide a sequence: Example of a calid sequence: 'atgctattttag' (without quotes): ")
# Example sequence:   atggaaattgtgctgacccaatctccgggcacactgagcttgtctccgggcgaacgtgcgacccttagctgcagagccagccagtcggtgtccagctcgtaccttaattggacctaccttacttggtatcaacagaaaccaggtcaagcacctcgcctgctgatttatggcgcctcttcacgtgccactggggtcccggatcgctttagcggctctggcagtggcaccgattttactctgaccatttcccgtctgaaaccggaagacttcgcggtgtactattgtcagcagtacaactccgtccctcttacctttggccaggggacgaaagtcgagattaaacgg


# Inference
aa_seq, codon_seq = pair_provider(seq)
prediction, tokens, attention_weights = reloaded(aa_seq)
predicted_codons = decode_codons(prediction)
predicted_aas = [str(Seq(x).translate()) for x in predicted_codons[1:-1]]
predicted_aas.insert(0, "[START]")
predicted_aas.append("[END]")

# This function shows the results of the translation
def show_results(seq1, seq2):
  seq1, seq2 = seq1[1:-1], seq2[1:-1] #remove [START] [STOP] to simplify alignment
  match = "|" * len(seq1[1])
  miss = "X" * len(seq1[1])
  alignment = [match if x == y else miss for x, y in zip(seq1, seq2)]
  misses = alignment.count(miss)
  similarity = round((1 - misses / len(alignment)) * 100, 2)
  print("")
  print(f'{"CODON ALIGNMENT" if len(match) == 3 else "AMINO ACID ALIGNMENT"}')
  print("#################################################################################################################")
  print(f'{"STARTING SEQUENCE" :25s}: {"".join(seq1[:-1]) if len(match) == 3 else "".join(seq2[:-1])} ')
  print("-----------------------------------------------------------------------------------------------------------------")
  print(f'{"Starting seq" :25s}: {" ".join(seq1)}, {len(seq1)}')
  print(f'{"Alignment" :25s}: {" ".join(alignment)}')
  print(f'{"Predicted seq" :25s}: {" ".join(seq2)}, {len(seq2)}')
  print(f'{"Stats" :25s}: Altered {"codons" if len(match) == 3 else "amino acids"}: {misses}, similarity {similarity} %')
  print("-----------------------------------------------------------------------------------------------------------------")
  print(f'{"RESULT" :25s}: {"".join(seq2[:-1])}')
  
  
  
def show_CAI(seq, ref_path="aa2codon/ecol.heg.fasta"):
  # codon adaptation index
  lst = []
  for record in SeqIO.parse(ref_path, "fasta"):
      lst.append(record.seq)
  return(CAI(seq, reference=lst))

cai1 = show_CAI(seq)
cai2 = show_CAI("".join(predicted_codons[1:-1]))


show_results(codon_seq.split(" "), predicted_codons)
show_results(aa_seq.split(" "), predicted_aas)
print(f"STARTING SEQUENCE CAI: {round(cai1, 2) * 100}")

print(f"OPTIMIZED SEQUENCE CAI: {round(cai2, 2) * 100}")

#plot_attention_weights(seq,
#                       tokens,
#                       attention_weights[0])



# visualization of sequence data


# Create a list with the order of the sequences for plotting

def makeordered(lista):
  lst = []
  n = 1
  for i in lista:
    if i == "[START]":
      n = n - 1
    lst.append(f"{(i, n)}")
    n = n + 1
  return lst

# Plot the attention weights for a specific attention head

def plot_heads(seq, tokens, attention):
  tokens = tokens
  seq = aa2id(seq)
  df = pd.DataFrame(np.array(attention))
  df.columns =  makeordered([label for label in decode_aas(seq.numpy())])
  df.index = makeordered([label for label in decode_codons(tokens[1:].numpy())])
  return df


# Plot the attention weights for all attention heads

def plot_headsit(attention_weights):
  img_seq = []
  for i in attention_weights:
    img_seq.append(plot_heads(aa_seq, tokens, i))
  return img_seq

# Plot the attention weights with Plotly

def plotter(plot_lst):
  n = 1
  for i in plot_lst:
    plot = px.imshow(i,
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title=f"Attention Head {n}")
    plot.show()
    n = n + 1
lst = plot_headsit(attention_weights[0])
plotter(lst)
import pickle

from CAI import CAI

## Stored vectorize layers
from_disk = pickle.load(open("/content/aa2codon/aa2id.pkl", "rb"))
aa2id = tf.keras.layers.TextVectorization.from_config(from_disk['config'])

aa2id.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
aa2id.set_weights(from_disk['weights'])

from_disk = pickle.load(open("/content/aa2codon/codon2id.pkl", "rb"))
codon2id = tf.keras.layers.TextVectorization.from_config(from_disk['config'])

codon2id.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
codon2id.set_weights(from_disk['weights'])

codon_voc = codon2id.get_vocabulary()
aa_voc = aa2id.get_vocabulary()

id2codon = {str(codon_voc.index(i)) : i for i in codon_voc}
id2aa = {str(aa_voc.index(i)) : i for i in aa_voc}

# Vector ---> codon and aa
def decode_codons(result):
  return [id2codon[str(key)] for key in np.array(result)]

def decode_aas(result):
  return [id2aa[str(key)] for key in np.array(result)]



# model
# Load the saved model
sijainti = "/content/aa2codon/model_aa2id"
print("Loading the trained model...")
reloaded = tf.saved_model.load(f'{sijainti}')
# Print a message to indicate that the model is being loaded
print("Ladattu:" , reloaded)

# Inference

# Returns a random sample pair of 
# amino acids and codons where the
# length of the amino acids is less
# than the specified length. Optional
def sample_pair(pairs, length):
  samples = [i for i in pairs if len(i[0]) < length]
  test_pair = samples[randrange(len(samples))]
  aa = test_pair[0]
  codon = test_pair[1]
  return aa, codon

# Aligns the predicted and ground truth sequences, and returns the accuracy of the prediction

def align(prediction, ground_truth):
  alignment = ["|||" if x == y else "XXX" for x, y in zip(prediction.split(" "), ground_truth.split(" "))]
  alignment = alignment[1:-1]
  counter = alignment.count("XXX")
  accuracy = counter / len(alignment)
  return alignment, accuracy, counter
  
def aa_align(prediction, ground_truth):
  alignment = ["|" if x == y else "X" for x, y in zip(prediction.split(" "), ground_truth.split(" "))]
  counter = alignment.count("X")
  accuracy = counter / len(alignment)
  return alignment, accuracy, counter

# Plots the attention mechanism for a single head from the given tensor

def plot_attention_head(in_tokens, translated_tokens, attention):
  # Plot one head from given tensor
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label for label in decode_codons(in_tokens.numpy())]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label for label in decode_codons(translated_tokens.numpy())]
  ax.set_yticklabels(labels)

def plot_attention_weights(sentence, translated_tokens, attention_heads):
  # Collects all heads as subplots
  in_tokens = aa2id(sentence)
  print(in_tokens)
  fig = plt.figure(figsize=(64, 32))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(in_tokens, translated_tokens, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()


def seq_check(seq):
  # Check if the input sequence is nucleotide or amino acid
  if any(c in 'atgc' for c in seq.lower()):  # Don't use str as a name.
    return "nt"
  elif any(c in 'randcqeghilkmfpstwyv*' for c in seq.lower()):
    return "aa"

  
def splitseq(string, length):
    # Split the input sequence into parts of given length
    return ' '.join(string[i:i+length] for i in range(0,len(string),length))

def pair_provider(seq):
  # Return the amino acid and codon pairs for the input sequence
  seq = seq.replace(" ", "")
  if seq_check(seq) == "nt":
    if len(seq) % 3 != 0:
      print("HUOM! Ei kolmella jaollinen sekvenssi!")
    seq = Seq(seq)
    aa = str(seq.translate())
    aa = splitseq(aa, 1)
    seq = str(seq)
    seq = splitseq(seq, 3)
  elif seq_check(seq) == "aa":
    return "[START] " + " ".join(list(seq.upper())) + " [END]", "N/A"
  return "[START] " + str(aa) + " [END]", "[START] " + str(seq).upper() + " [END]"
