import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
import torch
import torch_geometric


def test_initial_stack_imports() -> None:
    tensor = torch.tensor([1, 2, 3])
    array = np.array([1, 2, 3])
    frame = pd.DataFrame({"gene": ["TP53"]})
    graph = nx.Graph()

    assert tensor.sum().item() == 6
    assert array.sum() == 6
    assert frame.shape == (1, 1)
    assert graph.number_of_nodes() == 0
    assert sklearn.__version__
    assert matplotlib.__version__
    assert torch_geometric.__version__
