
########################################
# Compute Average on ABC Dataset
########################################

compute_average_on_abc_dataset:
	python compute_abc_data \
		--size 8192 \
		--batch_size 64 \
		--prompt_type "BABA"


########################################
# Run comparative inference: Model vs Abalted Model
########################################

test_circuit:
	python test_circuit.py


minimality:
	python minimality.py

completeness:
	python completeness.py \
		--ablation_set_type "random" \