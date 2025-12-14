from model import AblatableGPT2Model
import argparse
import torch
from ioi_dataset import IOIDataset
import os
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from compute_abc_data import compute_abc_data
import math
from tqdm import tqdm
import ast
import copy

##Our functions
from ioi_task import run_ioi_task

def config_compatibility(minimality_sets_config,circuit):
    
    minimality_sets = minimality_sets_config["minimality_sets"]
    heads = [ast.literal_eval(head) for head in minimality_sets.keys()]
    for head in heads:
        if head[1] not in circuit[str(head[0])]:
            return False

    for layer in circuit.keys():
        for head in circuit[layer]:
            if (int(layer),head) not in heads:
                return False

    return True

def build_minimality_ablation_configs(circuit_config,minimality_set,head_to_evaluate):
    head_to_evaluate = ast.literal_eval(head_to_evaluate)
    circuit_config_with = copy.deepcopy(circuit_config["attention_heads"])

    for head in minimality_set:
        circuit_config_with[str(head[0])].remove(head[1])

    circuit_config_without = copy.deepcopy(circuit_config_with)
    circuit_config_without[str(head_to_evaluate[0])].remove(head_to_evaluate[1])

    print(f"for head {head_to_evaluate}, \n minimality set without heads: {circuit_config_without} \n minimality set with heads: {circuit_config_with}")
    ablation_config_with = {
        "name" : f"{circuit_config['name']}_{head_to_evaluate}_minimality_set_with",
        "attention_heads" : circuit_config_with,
        "ablate_mlp" : False
    }

    ablation_config_without = {
        "name" : f"{circuit_config['name']}_{head_to_evaluate}_minimality_set_without",
        "attention_heads" : circuit_config_without,
        "ablate_mlp" : False
    }

    return ablation_config_with, ablation_config_without

def compute_minimality(args):

    size = args.size
    device = args.device

    #############################################################################################
    # Builds the IOI dataset
    #############################################################################################

    ioi_samples = IOIDataset(
        prompt_type=args.prompt_type,
        N=size,
        nb_templates=args.num_templates,
        seed = 0,
        template_idx=args.template_index
    )

    print("----------------------------------------------------------------------------------------------------")
    print("A few samples from the IOI dataset  : ")
    for sentence in ioi_samples.sentences[:5]:
        print(sentence)
    print("----------------------------------------------------------------------------------------------------")


    #############################################################################################
    # Loads the sums over the ABC dataset.
    #############################################################################################

    abc_path = os.path.join(args.checkpoints_folder, f"abc_{args.prompt_type}_")
    template_key = f"T={args.template_index}" if args.num_templates == 1 else f"N={args.num_templates}"
    abc_path += template_key

    template_O_position = ioi_samples.O_position

    if not os.path.isfile(abc_path):
        print("----------------------------------------------------------------------------------------------------")
        print("ABC data not found. Fall back to full computation.")
        compute_abc_data(args)

    abc_data = torch.load(abc_path, map_location = device)
    print(f"Successfully loaded previous ABC means from {abc_path}")


    #############################################################################################
    # Builds the dataset and dataloader
    #############################################################################################

    seq_len = ioi_samples.toks.shape[1]

    print("----------------------------------------------------------------------------------------------------")
    print("Sequence length : ", seq_len)

    ioi_inputs = ioi_samples.toks.long()[:, :seq_len - 1]

    ioi_labels = ioi_samples.toks.long()[:, seq_len - 1]
    ioi_O_labels = ioi_samples.toks.long()[:, template_O_position] ##This is only compatible with a single template running

    ioi_dataset = TensorDataset(ioi_inputs, ioi_labels, ioi_O_labels)

    loader = DataLoader(ioi_dataset, batch_size = args.batch_size, shuffle = False)

    #############################################################################################
    # Builds the model
    #############################################################################################^
    start_time = time.time()
    model = AblatableGPT2Model.from_pretrained("gpt2")
    print(f"Loaded GPT2 model: total time {time.time() - start_time:.2f} seconds")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction = "sum")

    #############################################################################################
    # Loads the ablation config
    #############################################################################################

    ablation_config_path = os.path.join(args.configs_folder, args.config)

    if os.path.isfile(ablation_config_path):
        ablation_config = json.load(open(ablation_config_path, "r"))
    else:
        raise(FileNotFoundError("The specified config file does not exist."))


    minimality_log = {}

    #############################################################################################
    # Loads the minimality sets config
    #############################################################################################

    minimality_sets_config_path = os.path.join(args.minimality_sets_configs_folder, args.config)
    if os.path.isfile(minimality_sets_config_path):
        minimality_sets_config = json.load(open(minimality_sets_config_path, "r"))
    else:
        raise(FileNotFoundError(f"The specified minimality config file does not exist {minimality_sets_config_path}."))

    try:
        assert config_compatibility(minimality_sets_config,ablation_config["attention_heads"])
    except:
        raise(ValueError("The minimality sets config is not compatible with the ablation config."))

    A = math.inf
    minimality_sets = minimality_sets_config["minimality_sets"]
    for head_to_evaluate in minimality_sets.keys():
        print(f'Dealing with head {head_to_evaluate} for minimality computation.')
        #############################################################################################
        # Builds the circuit
        #############################################################################################

        ablation_config_with, ablation_config_without = build_minimality_ablation_configs(ablation_config,minimality_sets[head_to_evaluate],head_to_evaluate)

        model.ablate(abc_data, ablation_config_without)
        minimality_circuit_name = ablation_config_without["name"] + "_" + head_to_evaluate + "_minimality_set_without"
        log_without = run_ioi_task(model, loader, criterion, device, args, size, template_key,minimality_circuit_name)

        model.ablate(abc_data, ablation_config_with)
        minimality_circuit_name = ablation_config_with["name"] + "_" + head_to_evaluate + "_minimality_set_with"
        log_with = run_ioi_task(model, loader, criterion, device, args, size, template_key,minimality_circuit_name)

        print("Minimality results for head ", head_to_evaluate)
        difference_without = log_without[args.prompt_type][template_key]["Logit difference"]
        difference_with = log_with[args.prompt_type][template_key]["Logit difference"]
        difference_in_difference = abs(difference_with - difference_without)
        print("Without head : ", difference_without)
        print("With head : ", difference_with)
        print("Difference in logit difference : ", difference_in_difference)

        A = min(A, difference_in_difference)
        print("A so far : ", A)

        minimality_log[head_to_evaluate] = {
            "Logit difference without head" : difference_without,
            "Logit difference with head" : difference_with,
            "Difference in logit difference" : difference_in_difference
        }
    # Saves the minimality results
    minimality_results_path = os.path.join(args.results_folder, f'minimality_{args.prompt_type}_{template_key}.json')
    with open(minimality_results_path, "w") as f:
        json.dump(minimality_log, f, indent=4)
    print(f"Saved minimality results to {minimality_results_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--results_folder", type = str, default = "results/minimality", help = "Results folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the IOI dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--num_templates", type = int, default = 1, help = "Number of different templates to use.")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'BABA', help = "Template to use.")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("--minimality_sets_configs_folder", type = str, default = "configs/minimality_sets", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi_paper_ablation.json", help = "Ablation config file")
    parser.add_argument("-i", "--template_index", type = int, default = 0, help = "Index of the template to use (if only one template is selected).")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    compute_minimality(args)


