import torch
import numpy as np

def get_previous_token_head_detection_pattern(tokens: torch.Tensor) -> torch.Tensor:
    '''
    Attention weights pattern for previous token head. 
    
    :param tokens: Tokens being fed to the attention head. 
    :type tokens: torch.Tensor
    :return: A 0-1 mask corresponding to the previous token pattern.
    :rtype: Any
    '''

    detection_pattern = torch.zeros(tokens.shape[-1], tokens.shape[-1])

    # Adds a diagonal of 1's below the main diagonal.
    detection_pattern[1:, :-1] = torch.eye(tokens.shape[-1] - 1)

    return torch.tril(detection_pattern)


# Duplicate token head
def get_duplicate_token_head_detection_pattern(tokens: torch.Tensor) -> torch.Tensor:
    '''
    Attention weights pattern for duplicate token head. 
    
    :param tokens: Tokens being fed to the attention head. 
    :type tokens: torch.Tensor
    :return: A 0-1 mask corresponding to the duplicate token pattern.
    :rtype: Any
    '''

    # [pos x pos]
    token_pattern = tokens.repeat(tokens.shape[-1], 1).numpy()

    # If token_pattern[i][j] matches its transpose, then token j and token i are duplicates.
    eq_mask = np.equal(token_pattern, token_pattern.T).astype(int)

    np.fill_diagonal(eq_mask, 0)  # Current token is always a duplicate of itself. Ignore that.
    detection_pattern = eq_mask.astype(int)
    return torch.tril(torch.as_tensor(detection_pattern).float())


# Induction head
def get_induction_head_detection_pattern(tokens: torch.Tensor) -> torch.Tensor:
    '''
    Attention weights pattern for induction head. 
    
    :param tokens: Tokens being fed to the attention head. 
    :type tokens: torch.Tensor
    :return: A 0-1 mask corresponding to the induction pattern.
    :rtype: Any
    '''

    duplicate_pattern = get_duplicate_token_head_detection_pattern(tokens)

    # Shift all items one to the right
    shifted_tensor = torch.roll(duplicate_pattern, shifts=1, dims=1)

    # Replace first column with 0's
    # we don't care about bos but shifting to the right moves the last column to the first,
    # and the last column might contain non-zero values.
    zeros_column = torch.zeros(duplicate_pattern.shape[0], 1)
    result_tensor = torch.cat((zeros_column, shifted_tensor[:, 1:]), dim=1)
    return torch.tril(result_tensor)

def get_s_inhibition_head_detection_pattern(tokens: torch.Tensor, template_O_position : int) -> torch.Tensor:

    O = tokens[:,template_O_position]

    matches = (tokens[0] == O).nonzero(as_tuple=True)[0]

    if len(matches) > 0:
        last_index = matches[-1].item()
    else:
        raise(Exception("The object is not repeated."))
    
    mask = np.zeros((tokens.size(1), tokens.size(1)))

    mask[-1, last_index] = 1

    return torch.tril(torch.as_tensor(mask).float())


def get_name_mover_head_detection_pattern(tokens: torch.Tensor, ioi_label) -> torch.Tensor:

    matches = (tokens[0] == ioi_label).nonzero(as_tuple=True)[0]

    if len(matches) > 0:
        first_index = matches[0].item()
    else:
        raise(Exception("The object is not repeated."))
    
    mask = np.zeros((tokens.size(1), tokens.size(1)))

    mask[-1, first_index] = 1

    return torch.tril(torch.as_tensor(mask).float())



def compute_head_attention_similarity_score(
    attention_pattern: torch.Tensor,  # [q_pos k_pos]
    detection_pattern: torch.Tensor,  # [seq_len seq_len] (seq_len == q_pos == k_pos)
    *,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: str) -> float:

    '''
    Compute the similarity between "attention_pattern" and "detection_pattern"
    
    :param attention_pattern: Lower triangular matrix (Tensor) representing the attention pattern of a particular attention head.
    :type attention_pattern: torch.Tensor
    :param detection_pattern: Lower triangular matrix (Tensor) representing the attention pattern we are looking for.
    :type detection_pattern: torch.Tensor
    :param exclude_bos: Description
    :type exclude_bos: `True` if the beginning-of-sentence (BOS) token should be omitted from comparison. `False` otherwise.
    :param exclude_current_token: Description
    :type exclude_current_token: `True` if the current token at each position should be omitted from comparison. `False` otherwise.
    :param error_measure:  "abs" for using absolute values of element-wise differences as the error measure. "mul" for using element-wise multiplication (legacy code).
    :type error_measure: str
    :return: Description
    :rtype: float
    '''

    # mul

    if error_measure == "mul":
        if exclude_bos:
            attention_pattern[:, 0] = 0
        if exclude_current_token:
            attention_pattern.fill_diagonal_(0)
        score = attention_pattern * detection_pattern
        return (score.sum() / attention_pattern.sum()).item()

    # abs

    abs_diff = (attention_pattern.squeeze(0) - detection_pattern).abs()

    selected_diffs = abs_diff[detection_pattern == 1.0]

    return 1 - round((selected_diffs.mean()).item(), 3)

    '''
    abs_diff = (attention_pattern - detection_pattern).abs()
    assert (abs_diff - torch.tril(abs_diff).to(abs_diff.device)).sum() == 0

    size = len(abs_diff)
    if exclude_bos:
        abs_diff[:, 0] = 0
    if exclude_current_token:
        abs_diff.fill_diagonal_(0)

    return 1 - round((abs_diff.mean() * size).item(), 3)
    '''