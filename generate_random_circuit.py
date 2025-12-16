def generate_random_circuit(n):
    model_layers =12
    model_heads_perlayer = 12
    model_heads = model_layers * model_heads_perlayer
    np.random.seed(42)  # For reproducibility
    heads = np.random.randint(0, model_heads, size=n).tolist()
    circuit = {str(i):[] for i in range(12)}
    for i in heads:

