import os
import torch
import json


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

