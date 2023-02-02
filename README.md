# aa2codon

You can run the tool from <a href="https://github.com/pyypyyy/aa2codon/blob/main/aa2codon_optimize.ipynb">here</a> directly in google colab.


You can also try training your own models from <a href="https://github.com/pyypyyy/aa2codon/blob/main/Codon_optimizer_train.ipynb">here</a>.



*********************************************************************


Optimisation tool based on transformers


This transformer-based tool is based on the work of Vaswani et al. al.: https://arxiv.org/abs/1706.03762

The model is built by modifying the example model provided by tensorflow to handle DNA sequences. The parameters of the model have also been significantly modified. The page explains well the operation and advantages of the transformer. In addition, the idea of an "accuracy matrix" is explained. https://www.tensorflow.org/text/tutorials/transformer

The accuracy of the model at the validation data set is 59.61% and at the protein level the accuracy is 100% at the validation set. Based on random experiments, I managed to make mutations by accidentally inserting a stop codon in the middle of a gene, and the model corrected it to some other codon.

The model exceeds the validation accuracy of previously reported models (52%, Fu et al. 2020; https://doi.org/10.1038/s41598-020-74091-z). Previous models were based on LSTM neural networks, which are outdated by now.

The model is trained by giving it amino acid sequences translated from genes, which it translates back into codons. The model has been trained with 12927 E. coli gene fragments with a maximum length of 450bp. The fragments were digested from 4242 genes. 10% of the sequences have been used for validation and the remaining 90% for training. The code for searching and editing genes for the exercise will be provided when I get it cleaned up.

The model metrics are stored here: https://tensorboard.dev/experiment/g4WHVqsBRfGPYnWNuesL1g/#scalars

2.2.2023

Added the Codon_optimizer_train.ipynb for training models.
