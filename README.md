# Python_Machine_Learning_App

PURPOSE:

1. Makes prediction whether particular instrument of the Stock Market will go up or down on particular day.

2. Makes prediction of price per particular instrument per particular day.
	
------------------------

DEVELOPMENT NOTES (to be ordered if the project is to be continued):

Machine Learning - Training Speed Comparison:

1. Test 1 (UNKNOWN SET):
	CPU (Mac Pro) 			7 ms / step
	Colab CPU 			14 ms / step
	Colab TPU 			17 ms / step

2. Test 2 (confirmed the same data trained):
	CPU (Mac Pro)			12 ms / step
	GPU (PlaidML; Mac Pro)		77 ms / step

3. Test 3 (same set as in test 2, but with optimizer SGD):
	GPU (tensorflow-metal)		76 ms / step

4. Test 4 (same set as in 2):
	Colab TPU			23 ms / step
	Colab GPU			8 ms / step
	Colab CPU (no accellerator) 	15 ms / step


Machine Learning App - Current Work Notes:

1. Make a list of potential parameters that may be optimised with Bayesian.

2. Is it okay if I use mean_absolute_error as loss function in training, but direction_prediction_result as measurement in Bayesian?

3. How to train Bayesian with many factors? Shall I try to optimise them all at once, or maybe few at a time?

4. Batch size vs epoch size?



