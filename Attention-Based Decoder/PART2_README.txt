# Part 2 - Attention-Based Decoder

#### Training Script

To Run the training script you should have the following files:
attention_training.py , utils.py

Script Path: MT_ex2\Attention-Based Decoder\attention_training.py

Parameters:
1. train.src - source train file
2. train.trg - target train file
3. dev.src - source dev file
4. dev.trg - target dev file

For example:
python3 attention_training.py ..\Data\train.src ..\Data\train.trg ..\Data\dev.src ..\Data\dev.trg

Output files:
encoder_file - best model checkpoint according to the BLEU score.
decoder_file - best model checkpoint according to the BLEU score.
dictionary_file - contains the vocabularies

#### Evaluation Script

To Run the evaluation script you should have the following files:
attention_evaluation.py , Utils.py

Script Path: MT_ex2\Attention-Based Decoder\attention_evaluation.py

parameters:
1. encoder_file
2. decoder_file
3. dictionary_file
4. test.src
5. test.trg

For example:
python3 attention_evaluation.py Output\encoder_file Output\decoder_file Output\dictionary_file ..\Data\test.src ..\Data\test.trg

Output files:
Test.pred