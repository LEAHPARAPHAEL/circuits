from model import AblatableGPT2Model
import argparse
import torch
from ioi_dataset import IOIDataset
import os
from torch.utils.data import TensorDataset, DataLoader




def compute_means(args):

    size = args.size
    device = args.device

    abc_samples = IOIDataset(
        prompt_type="ABC",
        N=size,
        nb_templates=4,
        seed = 0,
    )


    abc_means_filepath = os.path.join(args.checkpoints_folder, args.abc_means_filepath)

    if os.path.exists(abc_means_filepath):
        abc_means = torch.load(abc_means_filepath, map_location = device)

        print(f"Successfully loaded previous ABC means from {abc_means_filepath}")

        last_sample_index = abc_means["last_sample_index"]

    else:
        abc_means = None
        last_sample_index = 0


    seq_len = abc_samples.toks.shape[1]

    # We only care about the inputs since we just want to compute activations means

    abc_tokens = abc_samples.toks.long()[last_sample_index:size, :seq_len - 1].to(device)

    abc_dataset = TensorDataset(abc_tokens)

    loader = DataLoader(abc_dataset, batch_size = args.batch_size, shuffle = False)

    model = AblatableGPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()


    model.start_accumulation()
    print(f"Evaluating the model on the ABC dataset from index {last_sample_index} to {size}")
    with torch.no_grad():

        running_samples_count = 0

        for batch in loader:

            inputs = batch[0].to(device)
            
            model(inputs)

            running_samples_count += args.batch_size

            if running_samples_count % args.checkpoint_steps == 0:
                abc_means = model.update_accumulation_data(running_samples_count, abc_means_filepath, abc_means)
                running_samples_count = 0


    












if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-a", "--abc_means_filepath", type = str, default = "abc_means.pth", help = "Path to the saved means over the ABC dataset")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the ABC dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-c", "--checkpoint_steps", type = int, default = 1024, help = "Number of samples to evaluate between each checkpoint")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")



    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)

    compute_means(args)


