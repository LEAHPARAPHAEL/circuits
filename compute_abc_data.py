from model import AblatableGPT2Model
import argparse
import torch
from ioi_dataset import IOIDataset
import os
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
from tqdm import tqdm



def compute_abc_data(args : argparse.Namespace) -> None:

    '''
    Computes the sum of the outputs of each attention layer and MLP layer for the GPT2 model
    over the ABC dataset.
    
    :param args: Namespace containing the configuration, namely the batch size, the device, the 
                 size of the dataset...
    :type args: argparse.Namespace
    '''

    size = args.size
    device = args.device

    #############################################################################################
    # Builds the ABC samples
    #############################################################################################

    ioi_samples = IOIDataset(
        prompt_type=args.prompt_type,
        N=size,
        nb_templates=args.num_templates,
        seed = 0,
        template_idx=args.template_index
    )

    abc_samples = (
        ioi_samples.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )


    print("----------------------------------------------------------------------------------------------------")
    print("A few samples from the ABC dataset  : ")
    for sentence in abc_samples.sentences[:5]:
        print(sentence)
    print("----------------------------------------------------------------------------------------------------")


    #############################################################################################
    # Finds potential checkpoints
    #############################################################################################

    abc_path = os.path.join(args.checkpoints_folder, f"abc_{args.prompt_type}")
    abc_path += f"_T={args.template_index}" if args.num_templates == 1 else f"_N={args.num_templates}"
    
    if os.path.isfile(abc_path):
        print("Activations on this ABC template have already been computed.")
        return

    #############################################################################################
    # Builds the dataset and dataloader
    #############################################################################################

    seq_len = abc_samples.toks.shape[1]

    print("----------------------------------------------------------------------------------------------------")
    print("Sequence length : ", seq_len)

    abc_tokens = abc_samples.toks.long()[:size, :seq_len - 1].to(device)

    abc_dataset = TensorDataset(abc_tokens)

    loader = DataLoader(abc_dataset, batch_size = args.batch_size, shuffle = False)


    #############################################################################################
    # Builds the model
    #############################################################################################^

    model = AblatableGPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()


    #############################################################################################
    # Evaluation loop
    #############################################################################################

    model.accumulate()
    print("----------------------------------------------------------------------------------------------------")
    print(f"Evaluating the model on the ABC dataset for template {args.prompt_type} with {size} samples")
    with torch.no_grad():

        for batch in tqdm(loader):

            inputs = batch[0].to(device)
            
            model(inputs)

    abc_data = model.stop_accumulation()

    abc_data["size"] = size

    torch.save(abc_data, abc_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the sum of outputs of the gpt2 model over the ABC dataset")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the ABC dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--num_templates", type = int, default = 1, help = "Number of different templates to use.")
    parser.add_argument("-i", "--template_index", type = int, default = 0, help = "Index of the template to use (if only one template is selected).")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'BABA', help = "Templates to use.")



    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)

    compute_abc_data(args)


