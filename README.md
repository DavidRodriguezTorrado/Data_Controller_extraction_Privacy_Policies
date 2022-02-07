# Data_Controller_extraction_Privacy_Policies
Application of Named Entity Recognition (NER) to Privacy Policies in order to extract the Data Controller from a Privacy Policy

## Usage:
data_controller_extraction.py file contains the core functionality. Which is based on different available methods depending on the input format of the Privacy Policies. Each method return is always a Pandas Dataframe containing paragraphs with their corresponding output<br>
An example of usage can be found in test_data_controller_extraction.py

To use the trained SpaCy model, a guide depending on your pc properties can be found at: [SpaCy Guide](https://spacy.io/usage)

Example, for pip Package Manager on a MacOS computer:

`pip install -U pip setuptools wheel`<br>
`pip install -U spacy`<br>
`python -m spacy download en_core_web_trf`<br>


## Validation
Validation folder contains 109 privacy policies in txt format used to test the correctness functionality of the NER script (Evaluation). <br>
Those privacy policies were extracted from Google Play Store.

## Evaluation
Evaluation folder contains the csv result files that were manually annotated.<br>

- validation_109_bog_actual_final_results.csv<br>

This file contains the evaluation results of the classifier tool developed for classifying a paragraph as either containing Data Controller or not.<br>
The classification method is based on Bag of Words. This looks for certain strings or characters in text to make the classification.<br>
contains_data_controller column refers to boolean classification output, while actual_contains_controller is the manual annotation or ground thuth for the evaluation.

- spacy_109_bog_actual_results.csv<br>

This file contains the evaluation results of the spacy tool used for the extraction.<br>
The expected output from spacy implementation can be found as boolean in spacy_results_bool column. The spacy expected boolean output can be found in actual_contains_controller column (Any non controller paragraph should not be tested by Spacy). True of False in both columns are referred to finding the Data Controller in the results array found in spacy_results column.
If there is no expected Data Controller due to missclassification of the Bag of Words classifier, expected result is an empty array and spacy_result_bool is True when the empty array is returned.

- final_validation_109_bog.csv<br>

This file contains the pipelined evaluation results of the classification method implemented to check if a paragraph contains or not Data Controller.<br>
Pipelined metrics where obtained from comparing obtained output (expected_output column) with actual results (actual_contains_controller column, manually annotated).<br>

## Metrics
Pipelined classification + extraction results metrics (final_validation_109_bog.csv):<br>
<br>
Precision: 0.9279279279279279<br>
Recall: 0.8046875<br>
F1-score: 0.8619246861924686<br>
NVP: 0.9962591650456382<br>
Specificity: 0.9987998799879988<br>
F1-score-negative: 0.9975279047119635<br>
Accuracy: 0.9017436899939995<br>
Conf_matrix: [103, 25, 8, 6658]

SpaCy results metrics (final_validation_109_bog.csv):<br>
<br>
Precision: 1.0<br>
Recall: 0.9137931034482759<br>
F1-score: 0.9549549549549551<br>
NVP: 0.0<br>
Specificity: 0.0<br>
F1-score-negative: 0.0<br>
Accuracy: 0.9137931034482759<br>
Conf_matrix: [106, 10, 0, 0]
