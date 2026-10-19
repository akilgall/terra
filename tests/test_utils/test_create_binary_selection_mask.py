import unittest

import torch

from terra.utils import create_binary_selection_mask


class TestCreateBinarySelectionMask(unittest.TestCase):
    def test_basic_cell_case(self):
        tokens = torch.tensor([[1, 3, 2, 0, 0, 5, 4, 3, 1, 8],
                               [4, 5, 0, 0, 0, 2, 3, 0, 0, 0]])
        expected_mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.bool)
        computed_mask = create_binary_selection_mask(
            ns_tokens=tokens,
            seq_len_cell=5,
            selection_type='agg_cell',
            top_k=None,
            gene_id=None)
        print(computed_mask)
        self.assertTrue(torch.equal(expected_mask, computed_mask))

    def test_basic_neighborhood_case(self):
        tokens = torch.tensor([[1, 3, 2, 0, 0, 5, 4, 3, 1, 8],
                               [4, 5, 0, 0, 0, 2, 3, 0, 0, 0]])
        expected_mask = torch.tensor(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]],
            dtype=torch.bool)
        computed_mask = create_binary_selection_mask(
            ns_tokens=tokens,
            seq_len_cell=5,
            selection_type='agg_neighborhood',
            top_k=None,
            gene_id=None)
        print(computed_mask)
        self.assertTrue(torch.equal(expected_mask, computed_mask))

    def test_gene_cell_case(self):
        tokens = torch.tensor([[1, 3, 2, 0, 0, 5, 4, 3, 1, 8],
                               [4, 5, 0, 0, 0, 2, 3, 0, 0, 0]])
        expected_mask = torch.tensor(
            [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.bool)
        computed_mask = create_binary_selection_mask(
            ns_tokens=tokens,
            seq_len_cell=5,
            selection_type='gene_cell',
            top_k=None,
            gene_id=2)
        print(computed_mask)
        self.assertTrue(torch.equal(expected_mask, computed_mask))

    def test_gene_neighborhood_case(self):
        tokens = torch.tensor([[1, 3, 2, 0, 0, 5, 4, 3, 1, 8],
                               [4, 5, 3, 0, 0, 2, 3, 0, 0, 0]])
        expected_mask = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            dtype=torch.bool)
        computed_mask = create_binary_selection_mask(
            ns_tokens=tokens,
            seq_len_cell=5,
            selection_type='gene_neighborhood',
            top_k=None,
            gene_id=3)
        print(computed_mask)
        self.assertTrue(torch.equal(expected_mask, computed_mask))

    def test_top_k_cell_case(self):
        tokens = torch.tensor([[1, 3, 2, 0, 0, 5, 4, 3, 1, 8],
                               [4, 5, 0, 0, 0, 2, 3, 0, 0, 0]])
        expected_mask = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.bool)
        computed_mask = create_binary_selection_mask(
            ns_tokens=tokens,
            seq_len_cell=5,
            selection_type='agg_cell',
            top_k=2,
            gene_id=None)
        print(computed_mask)
        self.assertTrue(torch.equal(expected_mask, computed_mask))

    def test_top_k_graph_uniform_layout(self):
        tokens = torch.arange(1, 13).unsqueeze(0)  # [4][4][4]
        expected_mask = torch.tensor(
            [[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        computed_mask = create_binary_selection_mask(
            ns_tokens=tokens,
            seq_len_cell=4,
            selection_type='agg_graph',
            top_k=2,
            n_segments=3)
        self.assertTrue(torch.equal(expected_mask, computed_mask))

    def test_top_k_graph_asymmetric_layout(self):
        tokens = torch.arange(1, 13).unsqueeze(0)  # [6][3][3]
        expected_mask = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]], dtype=torch.bool)
        computed_mask = create_binary_selection_mask(
            ns_tokens=tokens,
            seq_len_cell=6,
            selection_type='agg_graph',
            top_k=2,
            n_segments=3,
            seq_len_neigh=3)
        self.assertTrue(torch.equal(expected_mask, computed_mask))

if __name__ == '__main__':
    unittest.main()
