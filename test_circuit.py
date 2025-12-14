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
from tqdm import tqdm

def run_ioi_task_on_model(model, loader, criterion, device, args, size, log, results_path, template_key,circuit_name = "GPT_2"):
    total_loss = 0
    total_correct_preds = 0
    total_logits_diff = 0
    for batch_idx, (batch_inputs, batch_labels,batch_O_labels) in enumerate(loader):
        with torch.no_grad():

            inputs = batch_inputs.to(device)
            labels = batch_labels.to(device)
            O_labels = batch_O_labels.to(device)
            
            outputs = model(inputs)

            all_logits = outputs.logits

            next_token_logits = all_logits[:, -1, :]
            batch_indices = torch.arange(labels.size(0)).to(device)
            logit_difference = next_token_logits[batch_indices, labels] - next_token_logits[batch_indices, O_labels]
            #print(f"Successfully computed logit difference between [Answer] and [Object] tokens : {logit_difference}")


            loss = criterion(next_token_logits, labels)
            current_loss = loss.item()
            total_loss += current_loss
            predictions = next_token_logits.argmax(dim=-1)

            correct_predictions = (predictions == labels).sum().item()
            total_correct_preds += correct_predictions

            logits_diff = logit_difference.sum().item()
            total_logits_diff += logits_diff

            if ((batch_idx + 1) * args.batch_size) % args.test_steps == 0:
                print(f"{circuit_name} after {(batch_idx + 1) * args.batch_size} samples | Loss : {current_loss / inputs.size(0):.4f} | Accuracy : {correct_predictions / inputs.size(0):.2f} | Logit difference : {logits_diff / inputs.size(0):.2f}")

    model_avg_loss = total_loss / size
    model_avg_correct_preds = total_correct_preds / size
    model_avg_logits_diff = total_logits_diff / size

    print(f"{circuit_name} final metrics after {size} samples | Loss : {model_avg_loss:.4f} | Accuracy : {model_avg_correct_preds:.2f} | Logit difference : {model_avg_logits_diff:.2f}")

    log[args.prompt_type][template_key]["Count"] = size
    log[args.prompt_type][template_key]["Loss"] = model_avg_loss
    log[args.prompt_type][template_key]["Accuracy"] = model_avg_correct_preds    
    log[args.prompt_type][template_key]["Logit difference"] = model_avg_logits_diff

    json.dump(log, open(results_path, "w"), indent = 4)

def test_ioi_circuit(args):

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
    # Gets or creates the result dict for the model
    #############################################################################################

    results_path = os.path.join(args.results_folder, f"gpt2.json")

    if os.path.isfile(results_path):
        log = json.load(open(results_path, "r"))
    else:
        log = {}

    if args.prompt_type not in log:
        log[args.prompt_type] = {}
    
    if template_key not in log[args.prompt_type]:
        log[args.prompt_type][template_key] = {}
        do_need_to_compute = True
    else:
        do_need_to_compute = False


    #############################################################################################
    # First evaluation loop for the whole model
    #############################################################################################

    print("----------------------------------------------------------------------------------------------------")
    print(f"Evaluating the entire GPT2 model on the IOI dataset with template {args.prompt_type} and {size} samples.")

    if do_need_to_compute:
        run_ioi_task_on_model(model, loader, criterion, device, args, size, log, results_path, template_key)
    else:
        print("Retrieving results from previous computations.")
        print(f'GPT2 final metrics after {log[args.prompt_type][template_key]["Count"]} samples | Loss : {log[args.prompt_type][template_key]["Loss"]:.4f} | Accuracy : {log[args.prompt_type][template_key]["Accuracy"]:.2f} | Logit difference : {log[args.prompt_type][template_key]["Logit difference"]:.2f}')


    #############################################################################################
    # Builds the circuit
    #############################################################################################

    ablation_config_path = os.path.join(args.configs_folder, args.config)

    if os.path.isfile(ablation_config_path):
        ablation_config = json.load(open(ablation_config_path, "r"))
    else:
        raise(FileNotFoundError("The specified config file does not exist."))

    model.ablate(abc_data, ablation_config)

    #############################################################################################
    # Gets or creates the result dict for the circuit
    #############################################################################################

    results_path = os.path.join(args.results_folder, f'{ablation_config["name"]}.json')

    if os.path.isfile(results_path):
        log = json.load(open(results_path, "r"))
    else:
        log = {}

    if args.prompt_type not in log:
        log[args.prompt_type] = {}
    
    if template_key not in log[args.prompt_type]:
        log[args.prompt_type][template_key] = {}
        do_need_to_compute = True

    else:
        do_need_to_compute = False


    #############################################################################################
    # Second evaluation loop for the circuit
    #############################################################################################

    print("----------------------------------------------------------------------------------------------------")
    print(f"Evaluating the IOI circuit on the IOI dataset with template {args.prompt_type} and {size} samples.")

    if do_need_to_compute:
        run_ioi_task_on_model(model, loader, criterion, device, args, size, log, results_path, template_key,ablation_config["name"])
    else:
        print("Retrieving results from previous computations.")
        print(f'{ablation_config["name"]} metrics on {log[args.prompt_type][template_key]["Count"]} samples | Loss : {log[args.prompt_type][template_key]["Loss"]:.4f} | Accuracy : {log[args.prompt_type][template_key]["Accuracy"]:.2f}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--results_folder", type = str, default = "results", help = "Results folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the IOI dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--num_templates", type = int, default = 1, help = "Number of different templates to use.")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'BABA', help = "Template to use.")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi.json", help = "Ablation config file")
    parser.add_argument("-i", "--template_index", type = int, default = 0, help = "Index of the template to use (if only one template is selected).")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    test_ioi_circuit(args)


