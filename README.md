### Randomized neural networks for preference learning with physiological data

Reference code for the paper:

`D. Bacciu,  M. Colombo, D. Morelli, D. Plans, Randomized neural networks for preference learning with physiological data, provisionally accepted for publication in the Neurocomputing journal (minor revision), 2017`

Should you use the code, please cite (provisionally):

`D. Bacciu,  M. Colombo, D. Morelli, D. Plans,  ELM Preference Learning for Physiological Data, Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN'17), i6doc.com, Louvain-la-Neuve, Belgium, 2017`


### Abstract
>The paper discusses the use of randomized neural networks to learn a complete ordering between samples of heart-rate variability data by relying solely on partial and subject-dependent information concerning pairwise relations between samples. We confront two approaches, i.e. Extreme Learning Machines and Echo State Networks, assessing the effectiveness in exploiting hand-engineered heart-rate variability features versus using raw beat-to-beat sequential data. Additionally, we introduce a weight sharing architecture and a preference learning error function whose performance is compared with a standard architecture realizing pairwise ranking as a binary-classification task. The models are evaluated on real-world data from a mobile application realizing a guided breathing exercises, using a dataset of over 54K exercising sessions. Results show how a randomized neural model processing information in its raw sequential form can outperform its vectorial counterpart, increasing accuracy in predicting the correct sample ordering by about 20%. Further, the experiments highlight the importance of using weight sharing architectures to learn smooth and generalizable complete orders induced by the preference relation.

### How to

Once placed the expected data in `data/` (see [data readme](data/README.md) )
just ran the script you're interested in, check the supported flags on top of the python file. For example `python esn.py --action=select --prefix=esn_test1`

Note: all the code had been ran using Tensorflow 1.0
