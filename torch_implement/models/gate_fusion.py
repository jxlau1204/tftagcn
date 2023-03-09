import torch 
import torch.nn as nn 
import torch.nn.functional as F


def kronecker_product(tensor1 : torch.Tensor, tensor2: torch.Tensor):
    tensor1 = tensor1.unsqueeze(2)
    tensor2 = tensor2.unsqueeze(1)
    return (tensor1 @ tensor2).reshape(tensor1.shape[0], -1)
    


class GataFusion(nn.Module):
    def __init__(self, tf_cnn_size, dense_capsule_size, pretrain_size, is_gate, is_fusion, hidden_size, out_size, dropout=0.5) -> None:
        super().__init__()
        self.tf_cnn_filter = nn.Sequential(
            nn.Linear(tf_cnn_size, hidden_size),
            nn.ReLU()
        )
        self.capsule_filter = nn.Sequential(
            nn.Linear(dense_capsule_size, hidden_size),
            nn.ReLU()
        )
        self.pretrain_filter = nn.Sequential(
            nn.Linear(pretrain_size, hidden_size),
            nn.ReLU()
        )
        self.is_gate = is_gate
        self.is_fusion = is_fusion
        if self.is_gate:
            self.tf_cnn_gate = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.Sigmoid()
            )
            self.capsule_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
            self.pretrain_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        representation_input_dim = hidden_size * 3
        if self.is_fusion:
            representation_input_dim += hidden_size ** 2 * 3
        self.representation_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(representation_input_dim, out_size),
            nn.ReLU()
        )
    def forward(self, tf_cnn_input, capsule_input, pretrain_input):
        log = {}
        tf_cnn_out = self.tf_cnn_filter(tf_cnn_input)
        capsule_out = self.capsule_filter(capsule_input)
        pretrain_out = self.pretrain_filter(pretrain_input)
        
        if self.is_gate:
            tf_cnn_gate = self.tf_cnn_gate(torch.cat([capsule_out, pretrain_out], -1))
            capsule_gate = self.capsule_gate(torch.cat([tf_cnn_out, pretrain_out], -1))
            pretrain_gate = self.pretrain_gate(torch.cat([tf_cnn_out, capsule_out], -1))
            
            tf_cnn_out = capsule_out * tf_cnn_gate
            capsule_out = capsule_out * capsule_gate
            pretrain_out = pretrain_out * pretrain_gate
        if self.is_fusion:
            tfcnn_densecap_kron = kronecker_product(tf_cnn_out, capsule_out)
            tfcnn_pretrain_kron = kronecker_product(tf_cnn_out, pretrain_out)
            densecap_pretrain_kron = kronecker_product(capsule_out, pretrain_out)
            data = torch.cat([tf_cnn_out, 
                              capsule_out, 
                              pretrain_out, 
                              tfcnn_densecap_kron, 
                              tfcnn_pretrain_kron,
                              densecap_pretrain_kron], -1)
        else:
            data = torch.cat([tf_cnn_out, 
                              capsule_out, 
                              pretrain_out], -1)
        output = self.representation_layer(data)
        return output, log
        
        
        
    



# def gate_fusion(tf_cnn_size, dense_capsule_size, pretrain_size, is_gate, is_fusion, hidden_size=100):
#     _tf_cnn = Input(shape=(tf_cnn_size,), dtype='float32', name='tf_cnn')
#     _tf_cnn_out = Dense(hidden_size, activation='relu', name='tf_cnn_out')(_tf_cnn)

#     _dense_capsule = Input(shape=(dense_capsule_size,), name='dense_capsule')
#     _dense_capsule_out = Dense(hidden_size, activation='relu', name='dense_capsule_out')(_dense_capsule)

#     _pretrain = Input(shape=(pretrain_size,), name="pretrain")
#     _pretrain_out = Dense(hidden_size, activation='relu', name='pretrain_out')(_pretrain)

#     if is_gate:
#         _tf_cnn_gate = Dense(hidden_size, activation="sigmoid", name='tf_cnn_gate')(
#             concatenate([_dense_capsule_out, _pretrain_out],
#                         axis=-1))
#         _dense_capsule_gate = Dense(hidden_size, activation="sigmoid", name='dense_capsule_gate')(
#             concatenate([_tf_cnn_out, _pretrain_out],
#                         axis=-1))
#         _pretrain_gate = Dense(hidden_size, activation="sigmoid", name='pretrain_gate')(
#             concatenate([_tf_cnn_out, _dense_capsule_out],
#                         axis=-1))
#         _tf_cnn_filtered = multiply([_tf_cnn_out, _tf_cnn_gate])
#         _dense_capsule_filtered = multiply([_dense_capsule_out, _dense_capsule_gate])
#         _pretrain_filtered = multiply([_pretrain_out, _pretrain_gate])

#     if is_fusion:
#         tfcnn_densecap_kron = Kronecker([_dense_capsule_filtered, _tf_cnn_filtered])
#         tfcnn_pretrain_kron = Kronecker([_pretrain_filtered, _tf_cnn_filtered])
#         densecap_pretrain_kron = Kronecker([_dense_capsule_filtered, _pretrain_filtered])
#         # images_leaves_texts_kron = Kronecker([images_out, texts_leaves_kron])
#         # datas = [texts_out, images_out, leaves_out, texts_images_kron,
#         #          texts_leaves_kron, images_leaves_kron, images_leaves_texts_kron]
#         datas = [_tf_cnn_out, _dense_capsule_out, _pretrain_out, tfcnn_densecap_kron,
#                  tfcnn_pretrain_kron, densecap_pretrain_kron]
#     else:
#         datas = [_tf_cnn_out, _dense_capsule_out, _pretrain_out]

#     cat_data = concatenate(datas)
#     cat_hidden = Dropout(0.5)(cat_data)

#     cat_out = Dense(1024, activation="relu")(cat_hidden)
#     # cat_hidden = Dropout(0.5)(cat_hidden)
#     # # cat_hidden = Dense(300, activation="tanh")(cat_hidden)
#     # # cat_hidden = Dropout(0.5)(cat_hidden)
#     #
#     # cat_out = Dense(2, activation='sigmoid', name='cat_out')(cat_hidden)

#     _model = Model(inputs=[_tf_cnn, _dense_capsule, _pretrain], outputs=cat_out)
#     return _model