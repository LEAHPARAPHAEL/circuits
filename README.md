## Testing the circuit

To test our code, the simplest is to launch :

    python test_circuit.py 


## Templates

The template fom which to build the examples can be specified using the command line argument:

    python test_circuit.py -p BABA -i 6 

Where -p denotes the type of template (should be "BABA" or "ABBA"), and -i the index of the template,
which should be between 0 and 15. 


## Configuration file (defaults to paper_ablation.json)

The configuration of the circuit can be specified using the -c argument. A configuration file is typically located
in the configs folder, and contains the details of which heads to keep. If you want to test a different circuit, please
refer to the same syntax as in the file configs/paper_ablation.json. Then, assuming you create a new config file
named test.json, you can test it using :

    python test_circuit.py -p BABA -i 6 -c test.json

Warning : for most of the templates, the results have already been computed and will be fetched 
automatically. With indices higher than 6, the code should try to compute new results.


## Details of the implementation in test_circuit

This will perform, in this order, several steps :

1. Computes the sum of activations over the ABC dataset for the specified template.
2. Tests the GPT2 base model on several metrics over this template (see ioi_task.py for more details).
3. Tests the circuit on the same metrics.
4. Calls the function detect_heads from the file heads_detection.py to interpret the behaviour of the
   attention heads of the circuit.


## Results

The results can be found in the following locations :

1. The sum of activations per template are stored in the checkpoints/ directory.
2. The metrics for GPT2 are written in the file results/gpt2.json.
3. The metrics for a circuit named "my_circuit" (which is specified in the config file) are written 
   in the file results/my_circuit.json.
4. The results of the attention heads analysis are stored in the file results/heads/my_circuit.json.
   This will also generate plots, if no plots existed before for this circuit, in the directory plots,
   with the naming convention : plots/my_circuit_{layer}_{head_index}. 
   To not have too many plots, and because the behaviour of an attention head doesn't change a lot between
   templates, these plots are not recomputed for every template, but only once per circuit.
