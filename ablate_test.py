from model import AblatableGPT2Model
import argparse
import torch
from ioi_dataset import IOIDataset
import os
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
import time
=======
import json

>>>>>>> e9e9417e45c6cfe54548119eebf3c25b588ff0ea

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

    full_path = f"{args.abc_data_path}_{args.prompt_type}_"
    template_key = f"T={args.template_index}" if args.num_templates == 1 else f"N={args.num_templates}"
    full_path += template_key
    potential_filepaths = glob(f"{full_path}_*", root_dir = args.checkpoints_folder)

    if potential_filepaths:
        abc_data_path = os.path.join(args.checkpoints_folder, potential_filepaths[0])
        print(f"Successfully loaded previous ABC means from {abc_data_path}")

    else:
        raise(FileNotFoundError("No ABC data file was found ! One is required to perform ablation. Please execute" \
        "the file compute_abc_data.py first. "))


    #############################################################################################
    # Builds the dataset and dataloader
    #############################################################################################

    seq_len = ioi_samples.toks.shape[1]

    print("----------------------------------------------------------------------------------------------------")
    print("Sequence length : ", seq_len)

    ioi_inputs = ioi_samples.toks.long()[:, :seq_len - 1]

    ioi_labels = ioi_samples.toks.long()[:, seq_len - 1]

    ioi_dataset = TensorDataset(ioi_inputs, ioi_labels)

    loader = DataLoader(ioi_dataset, batch_size = args.batch_size, shuffle = False)

    #############################################################################################
    # Builds the model
<<<<<<< HEAD
    #############################################################################################^
    start_time = time.time()
=======
    #############################################################################################

>>>>>>> e9e9417e45c6cfe54548119eebf3c25b588ff0ea
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
        total_loss = 0
        total_correct_preds = 0
        for batch_idx, (batch_inputs, batch_labels) in enumerate(loader):
            with torch.no_grad():

                inputs = batch_inputs.to(device)
                labels = batch_labels.to(device)
                
                outputs = model(inputs)

                all_logits = outputs.logits

                next_token_logits = all_logits[:, -1, :]

                loss = criterion(next_token_logits, labels)
                current_loss = loss.item()
                total_loss += current_loss

                predictions = next_token_logits.argmax(dim=-1)

                correct_predictions = (predictions == labels).sum().item()

                total_correct_preds += correct_predictions

                if ((batch_idx + 1) * args.batch_size) % args.test_steps == 0:
                    print(f"GPT2 after {(batch_idx + 1) * args.batch_size} samples | Loss : {current_loss / inputs.size(0):.4f} | Accuracy : {correct_predictions / inputs.size(0):.2f}")

        model_avg_loss = total_loss / size
        model_avg_correct_preds = total_correct_preds / size

        print(f"GPT2 final metrics after {size} samples | Loss : {model_avg_loss:.4f} | Accuracy : {model_avg_correct_preds:.2f}")

        log[args.prompt_type][template_key]["Count"] = size
        log[args.prompt_type][template_key]["Loss"] = model_avg_loss
        log[args.prompt_type][template_key]["Accuracy"] = model_avg_correct_preds    

        json.dump(log, open(results_path, "w"), indent = 4)

    else:
        print("Retrieving results from previous computations.")
        print(f'GPT2 final metrics after {log[args.prompt_type][template_key]["Count"]} samples | Loss : {log[args.prompt_type][template_key]["Loss"]:.4f} | Accuracy : {log[args.prompt_type][template_key]["Accuracy"]:.2f}')



    #############################################################################################
    # Builds the circuit
    #############################################################################################

    ablation_config_path = os.path.join(args.configs_folder, args.config)

    model.ablate(abc_data_path, ablation_config_path, device)
    

    #############################################################################################
    # Gets or creates the result dict for the circuit
    #############################################################################################

    results_path = os.path.join(args.results_folder, f"{model.circuit_name}.json")

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

        total_loss = 0
        total_correct_preds = 0
        for batch_idx, (batch_inputs, batch_labels) in enumerate(loader):

            with torch.no_grad():

                inputs = batch_inputs.to(device)
                labels = batch_labels.to(device)
                
                outputs = model(inputs)

                all_logits = outputs.logits

                next_token_logits = all_logits[:, -1, :]

                loss = criterion(next_token_logits, labels)
                current_loss = loss.item()
                total_loss += current_loss

                predictions = next_token_logits.argmax(dim=-1)

                correct_predictions = (predictions == labels).sum().item()

                total_correct_preds += correct_predictions

                if ((batch_idx + 1) * args.batch_size) % args.test_steps == 0:
                    print(f"IOI circuit after {(batch_idx + 1) * args.batch_size} samples | Loss : {current_loss / inputs.size(0):.4f} | Accuracy : {correct_predictions / inputs.size(0):.2f}")

        circuit_avg_loss = total_loss / size
        circuit_avg_correct_preds = total_correct_preds / size

        print(f"IOI circuit final metrics after {size} samples | Loss : {circuit_avg_loss:.4f} | Accuracy : {circuit_avg_correct_preds:.2f}")

        log[args.prompt_type][template_key]["Count"] = size
        log[args.prompt_type][template_key]["Loss"] = circuit_avg_loss
        log[args.prompt_type][template_key]["Accuracy"] = circuit_avg_correct_preds    

        json.dump(log, open(results_path, "w"), indent = 4)

    else:
        print("Retrieving results from previous computations.")
        print(f'{model.circuit_name} final metrics after {log[args.prompt_type][template_key]["Count"]} samples | Loss : {log[args.prompt_type][template_key]["Loss"]:.4f} | Accuracy : {log[args.prompt_type][template_key]["Accuracy"]:.2f}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--results_folder", type = str, default = "results", help = "Results folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-a", "--abc_data_path", type = str, default = "abc_data", help = "Path to the saved sums over the ABC dataset")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the IOI dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--num_templates", type = int, default = 1, help = "Number of different templates to use.")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'BABA', help = "Template to use.")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi.json", help = "Ablation config file")
    parser.add_argument("--test_steps", type = int, default = 1024, help = "Number of samples between each print in the console.")
    parser.add_argument("-i", "--template_index", type = int, default = 0, help = "Index of the template to use (if only one template is selected).")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    test_ioi_circuit(args)


