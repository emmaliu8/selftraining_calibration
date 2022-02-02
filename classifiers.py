from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        return self.fc(text)
