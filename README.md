# About
* This repository attempts to generate titles from arXiv paper abstracts via abstractive text summarization.
* The paper used to model summarization is [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304).
* The dataset used is the [arXiv](https://www.kaggle.com/Cornell-University/arxiv) dataset from Kaggle, from which abstracts and titles are extracted.

## Data and Models
* Make two folders with name ```data``` and ```models``` in the current directory.
* Place ```abstracts.pkl``` and ```titles.pkl```, which are just list of corresponding sentences stored in pickle format, in **data** folder.
* Place glove embeddings in the **data** folder.
* Checkpoints will be saved in the **models** folder.

## Results
1. **ABSTRACT**: We discuss how to construct shift-invariant probability measures over the space of bisequences of symbols, and how to describe such measures in terms of block probabilities. We then define cellular automata as maps in the space of measures and discuss orbits of shift-invariant probability measures under these maps. Subsequently, the local structure approximation is discussed as a method to approximate orbits of Bernoulli measures under the action of cellular automata. The final sections presents some known examples of cellular automata, both deterministic and probabilistic, for which elements of the orbit of the Bernoulli measure (probabilities of short blocks) can be determined exactly.
	1. **Gold Title**: Orbits of Bernoulli Measures in Cellular Automata
	1. **Generated title**: Shift-Invariant Probability Measures for Cellular Automata
1. **ABSTRACT**: We review three methods of counting abelian orbifolds of the form C^3/Gamma which are toric Calabi-Yau (CY). The methods include the use of 3-tuples to define the action of Gamma on C^3, the counting of triangular toric diagrams and the construction of hexagonal brane tilings. A formula for the partition function that counts these orbifolds is given. Extensions to higher dimensional orbifolds are briefly discussed.
	1. **Gold Title**: An Introduction to Counting Orbifolds
	1. **Generated title**: Counting Abelian Orbifolds of the form C^3/Gamma
1. **ABSTRACT**: 'We consider the construction of generic spherically symmetric thin-shell traversable wormhole spacetimes in standard general relativity. By using the cut-and-paste procedure, we comprehensively analyze the stability of arbitrary spherically symmetric thin-shell wormholes to linearized spherically symmetric perturbations around static solutions. While a number of special cases have previously been dealt with in scattered parts of the literature, herein we take considerable effort to make the analysis as general and unified as practicable. We demonstrate in full generality that stability of the wormhole is equivalent to choosing suitable properties for the exotic material residing on the wormhole throat.
	1. **Gold Title**: Generic spherically symmetric dynamic thin-shell traversable wormholes in standard general relativity
	1. **Generated title**: Spherically symmetric thin-shell wormhole in standard general relativity

## Usage
* Clone the repository and follow the steps in the above sub-heading.
* Run ```python main.py train [load_path]``` for training the model with optional previous weights.
* Run ```python main.py val load_path``` for evaluating a previous model on validation set.
* Run ```python main.py test load_path path_to_test_file``` for generating titles on given test abstracts.

## Notes
* For learning rate, epochs etc. check the ```train.py``` file.
* The paper uses Rouge-L as RL reward, instead of that, here Rouge-1 + Rouge-2 + Rouge-L is used, which seemed to be more stable and gave better score on validation set.

## References
* https://arxiv.org/abs/1705.0430
* https://arxiv.org/abs/1704.04368
* https://arxiv.org/abs/1612.00563
* https://github.com/rohithreddy024/Text-Summarizer-Pytorch