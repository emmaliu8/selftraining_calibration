from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, embed_dim, num_class, initrange=0.5, num_layers = 1):
        super(TextClassificationModel, self).__init__()
        self.embed_dim = embed_dim 
        self.num_class = num_class 
        self.initrange = initrange
        self.fc = nn.Linear(embed_dim, num_class)
        # self.init_weights(initrange)

    def init_weights(self, initrange):
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        return self.fc(text)
