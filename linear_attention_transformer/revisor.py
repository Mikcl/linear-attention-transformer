from torch import nn

class Revisor(nn.Module):
    """Revisor network is a generic concept but will make an implementation of one here.

    The concept is to reuse the embeddings of the Network it is used within and 
    have its own disconnected process for 'revising' these embeddings by learning to
    predict the next embedding.


    Parameters
    ----------
    embedding : Embedding - shared from the network.
    """
    def __init__(
        self,
        token_embedding: nn.Embedding,
        pos_embedding: nn.Module,
        layer_pos_embedding: nn.Module,
        # TODO - Pass Config Params
    ):
        super().__init__()

        self.num_embeddings = token_embedding.num_embeddings
        self.embedding_dim  = token_embedding.embedding_dim

        assert self.embedding_dim
        assert self.num_embeddings

        self.token_embedding = token_embedding
        self.pos_embedding = pos_embedding
        self.layer_pos_embedding = layer_pos_embedding

        self.hidden_dim = 4
        self.layers     = 1
        self.dense_dim  = 32

        # Generator Model Used - could be anything E.G Transformer
        # TODO - Make it LinearAttentionTransformer(revisor=False)
        self.model = nn.LSTM(
                self.embedding_dim, self.hidden_dim, batch_first=True, num_layers=self.layers
        )
        # Extension of self.model, could be encapsulated
        self.linear_1 = nn.Linear(self.hidden_dim, self.dense_dim)

        self.linear_token = nn.Linear(self.dense_dim, self.num_embeddings)

        self.linear_vector = nn.Linear(self.dense_dim, self.embedding_dim)


    def forward(
        self,
        x
    ):
        x = self.token_embedding(x)
        x = x + self.pos_emb(x).type(x.type())

        # TODO - use when using a transformer
        layer_pos_emb = self.layer_pos_emb(x)

        _, (h, c) = self.model(x)  # (n_layers, n_samples, hidden_dim)

        h_mean = h.mean(dim=0)  # (n_samples, hidden_dim)
        x = self.linear_1(h_mean)  # (n_samples, dense_dim)

        logits = self.linear_token(x)  # (n_samples, vocab_size)

        vectors = self.linear_vector(x) # (n_samples, embedding_dim)  - learning to predict a new "non-existent" 'token'

        return logits, vectors


class RevisorGeneric(nn.Module):
    """Reviser network is a generic concept but will make an implementation of one here.

    The concept is to reuse the embeddings of the Network it is used within and 
    have its own disconnected process for 'revising' these embeddings by learning to
    predict the next embedding.


    Parameters
    ----------
    embedding : Embedding - shared from the network.
    """
    def __init__(
        self,
        embedding_module: nn.Module,
        num_embeddings: int,
        embedding_dim: int,
        # TODO - Pass Config Params
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim

        assert self.embedding_dim
        assert self.num_embeddings

        self.embedding_module = embedding_module

        self.hidden_dim = 4
        self.layers     = 1
        self.dense_dim  = 32

        # Generator Model Used - could be anything E.G Transformer
        # TODO - Make it LinearAttentionTransformer(revisor=False)
        self.model = nn.LSTM(
                self.embedding_dim, self.hidden_dim, batch_first=True, num_layers=self.layers
        )
        # Extension of self.model, could be encapsulated
        self.linear_1 = nn.Linear(self.hidden_dim, self.dense_dim)

        self.linear_token = nn.Linear(self.dense_dim, self.num_embeddings)

        self.linear_vector = nn.Linear(self.dense_dim, self.embedding_dim)


    def forward(
        self,
        x
    ):
        x = self.embedding_module(x)

        _, (h, c) = self.model(x)  # (n_layers, n_samples, hidden_dim)

        h_mean = h.mean(dim=0)  # (n_samples, hidden_dim)
        x = self.linear_1(h_mean)  # (n_samples, dense_dim)

        logits = self.linear_token(x)  # (n_samples, vocab_size)

        vectors = self.linear_vector(x) # (n_samples, embedding_dim)  - learning to predict a new "non-existent" 'token'

        return logits, vectors