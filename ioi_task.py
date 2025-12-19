import os
import torch
import json
import time

from ioi_dataset import IOIDataset
from torch.utils.data import TensorDataset, DataLoader
from model import AblatableGPT2Model
import torch.nn as nn

def setup_ioi_task(args):
    #############################################################################################
    # Builds the IOI dataset
    #############################################################################################

    ioi_samples = IOIDataset(
        prompt_type=args.prompt_type,
        N=args.size,
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

    abc_data = torch.load(abc_path, map_location = args.device)
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
    model.to(args.device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction = "sum")
    return model, abc_data, loader, criterion, template_key



def setup_dictionary(args, template_key,circuit_name):
    results_path = os.path.join(args.results_folder, f'{circuit_name}.json')

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
    return log, results_path, do_need_to_compute

def run_ioi_task(model, loader, criterion, device, args, size, template_key,circuit_name):

    log, results_path, do_need_to_compute = setup_dictionary(args, template_key, circuit_name)

    print("----------------------------------------------------------------------------------------------------")
    print(f"Evaluating {circuit_name} on the IOI dataset with template {args.prompt_type} and {size} samples.")

    if do_need_to_compute:
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

        model_avg_loss = total_loss / size
        model_avg_correct_preds = total_correct_preds / size
        model_avg_logits_diff = total_logits_diff / size

        #This will be printed when dic is reloaded
        #print(f"{circuit_name} final metrics after {size} samples | Loss : {model_avg_loss:.4f} | Accuracy : {model_avg_correct_preds:.2f} | Logit difference : {model_avg_logits_diff:.2f}")

        log[args.prompt_type][template_key]["Count"] = size
        log[args.prompt_type][template_key]["Loss"] = model_avg_loss
        log[args.prompt_type][template_key]["Accuracy"] = model_avg_correct_preds    
        log[args.prompt_type][template_key]["Logit difference"] = model_avg_logits_diff

        json.dump(log, open(results_path, "w"), indent = 4)
    else:
        print("Retrieving results from previous computations.")
    print(f'{circuit_name} metrics on {log[args.prompt_type][template_key]["Count"]} samples | Loss : {log[args.prompt_type][template_key]["Loss"]:.4f} | Accuracy : {log[args.prompt_type][template_key]["Accuracy"]:.2f} | Logit difference : {log[args.prompt_type][template_key]["Logit difference"]:.2f}"')
    return log

def run_comparison_on_two_circuits(circuit_1, circuit_2, model, abc_data, loader, criterion, device, args, size, template_key,circuit_name):
    #circuit1 use to be ablated model config
    #circuit2 use to be ablated circuit config

    #############################################################################################
    # Run the IOI task on the ablated model 
    #############################################################################################^A

    model.ablate(abc_data, circuit_1)
    log1 = run_ioi_task(model, loader, criterion, device, args, size, template_key, circuit_name=circuit_1["name"])

    #############################################################################################
    # Run the IOI task on the ablated circuit 
    #############################################################################################^A

    model.ablate(abc_data, circuit_2)
    log2 = run_ioi_task(model, loader, criterion, device, args, size, template_key, circuit_name=circuit_2["name"])

    #############################################################################################
    # Store the results of this random ablation
    #############################################################################################^A

    return log1,log2 



