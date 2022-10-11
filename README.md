<html>
<body>
<h2>Open-Domain Aspect-Opinion Co-Mining with Double-Layer Span Extraction</h2>

The supervised extraction methods achieve state-of-the-art performance but require large-scale human-annotated training data. Thus, they are restricted for open-domain tasks due to the lack of training data. We propose an Open-Domain Aspect-Opinion Co-Mining (ODAO) method with Double-Layer span extraction framework to overcome this issue and simultaneously mine aspect terms, opinion terms, and their correspondance in joint model.

<h3> Dataset </h3>

The experiments are conducted on SemEval 14, 15, 16 restaurant and SemEval 14 laptop datasets.
The original dataset can be found in the data/original_data folder.

<h3> Pre-processing </h3>
Data pre-processing includes four steps:
<ul>
<li>Format Data: The original file is processed and stored in a dictionary.
Run pre-processing/format_data.py to perform this step.</li>
<li>Weak Label Generator: In this step, the CoreNLP dependency parser is executed
to generate the weak labels. 
Download the CORENLP jar files from https://stanfordnlp.github.io/CoreNLP/download.html.
Place stanford-corenlp-4.0.0.jar and stanford-corenlp-4.0.0-models.jar in dependency_parser folder and run the following command from the folder.

```java -Xmx8g -XX:-UseGCOverheadLimit -XX:MaxPermSize=1024m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9015  -port 9015 -timeout 1500000```

Once the process is running on port 9015, run pre-processing/pseudo_labels.py to generate weak labels.
</li>
<li>Split Data: Once the weak labels are generated for the original train set. This step
splits the train set into pseudo train (reviews for which the weak label generator has identified both aspect term and opinion term)
and pseudo test (otherwise). The pseudo test is used for prediction as part of self-training.
Run pre-processing/split_data.py to execute this step.
</li>
<li>Get Pairs: This step processes the pseudo train set to format it for training. Run pre-processing/get_pairs.py to execute this step.
</li>
</ul>

<h3>Training</h3>
<ul>
<li>Run training/train.py to train the model.</li>
</ul>

<h3>Citation</h3>
Kindly cite our paper

```@inproceedings{10.1145/3534678.3539386,
author = {Chakraborty, Mohna and Kulkarni, Adithya and Li, Qi},
title = {Open-Domain Aspect-Opinion Co-Mining with Double-Layer Span Extraction},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539386},
doi = {10.1145/3534678.3539386},
abstract = {The aspect-opinion extraction tasks extract aspect terms and opinion terms from reviews. The supervised extraction methods achieve state-of-the-art performance but require large-scale human-annotated training data. Thus, they are restricted for open-domain tasks due to the lack of training data. This work addresses this challenge and simultaneously mines aspect terms, opinion terms, and their correspondence in a joint model. We propose an Open-Domain Aspect-Opinion Co-Mining (ODAO) method with a Double-Layer span extraction framework. Instead of acquiring human annotations, ODAO first generates weak labels for unannotated corpus by employing rules-based on universal dependency parsing. Then, ODAO utilizes this weak supervision to train a double-layer span extraction framework to extract aspect terms (ATE), opinion terms (OTE), and aspect-opinion pairs (AOPE). ODAO applies canonical correlation analysis as an early stopping indicator to avoid the model over-fitting to the noise to tackle the noisy weak supervision. ODAO applies a self-training process to gradually enrich the training data to tackle the weak supervision bias issue. We conduct extensive experiments and demonstrate the power of the proposed ODAO. The results on four benchmark datasets for aspect-opinion co-extraction and pair extraction tasks show that ODAO can achieve competitive or even better performance compared with the state-of-the-art fully supervised methods.},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {66–75},
numpages = {10},
keywords = {review analysis, natural language processing, data mining},
location = {Washington DC, USA},
series = {KDD '22}
}
```
</body>
</html>