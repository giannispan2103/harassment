from torch.nn import Module
import torch
from torch.nn.functional import relu
from .modules import PretrainedEmbeddingLayer, CellLayer, MultiLayerPerceptron, LastState, SequentialModel, ConcatenationLayer


class VanillaRnn(Module):
    def __init__(self, embeddings,
                 trainable_embeddings=True,
                 embeddings_dropout=0.0,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=2,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param embeddings_dropout: dropout of the embeddings layer
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(VanillaRnn, self).__init__()
        self.input_list = ['text']
        self.name = "VanillaRnn"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer_harassment = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.decision_layer_sexual = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                              init_size=large_size,
                                                              out_size=1,
                                                              dropout=top_mlp_dropout,
                                                              inner_activation=top_mlp_activation,
                                                              outer_activation=top_mlp_outer_activation)
        self.decision_layer_physical = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                              init_size=large_size,
                                                              out_size=1,
                                                              dropout=top_mlp_dropout,
                                                              inner_activation=top_mlp_activation,
                                                              outer_activation=top_mlp_outer_activation)
        self.decision_layer_indirect = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                              init_size=large_size,
                                                              out_size=1,
                                                              dropout=top_mlp_dropout,
                                                              inner_activation=top_mlp_activation,
                                                              outer_activation=top_mlp_outer_activation)
        self.last_state = LastState(large_size, large_size)

        self.seq = SequentialModel([self.word_embedding_layer, self.cell, self.last_state])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        representation = self.seq(x)
        out_harassment = self.decision_layer_harassment(representation)
        out_indirect = self.decision_layer_indirect(representation)
        out_sexual = self.decision_layer_sexual(representation)
        out_physical = self.decision_layer_physical(representation)
        out = torch.cat([out_harassment, out_sexual, out_physical, out_indirect], dim=1)
        return out
