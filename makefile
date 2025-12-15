
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
	python minimality.py \
		-t 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

completeness:
	python completeness.py \
		--probabilities 0.1 0.3 \
		--num_sets 5


########################################
# SCRIPTS
########################################
job_name = paper_circuit_minimality_all

minimality_all:
	sbatch --job-name=$(job_name) \
		--output=logs/$(job_name).out \
		--error=logs/$(job_name).err \
		scripts/minimality_all.sh

c_job_name = paper_circuit_completeness_10_probes
completeness_multiple_probablities:
	sbatch --job-name=$(c_job_name) \
		--output=logs/$(c_job_name).out \
		--error=logs/$(c_job_name).err \
		scripts/completeness_10_probes.sh
