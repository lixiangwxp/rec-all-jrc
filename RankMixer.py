import tensorflow as tf 
from tensorflow.keras.layers import *

class SemanticTokenization(tf.keras.layers.Layer):
    def __init__(self, num_T, num_D, **kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.num_D = num_D
        self.dense_layers = [Dense(num_D, activation='linear') for _ in range(num_T)]
        
        
    def call(self, x):
        x = tf.split(x, self.num_T, axis=-1)  # (B, num_T, D/num_T)
        x = [layer(x[i]) for i, layer in enumerate(self.dense_layers)]
        return  tf.stack(x, axis=1) # (B, num_T, num_D)
        
        
        
class TokenMixer(Layer):
    def __init__(self,num_T,num_D,num_H,**kwargs):
        super().__init__(**kwargs)
        self.num_T = num_T
        self.num_D = num_D
        self.num_H = num_H
        self.d_k = num_D//num_H
    
    def call(self,x):
        x = tf.reshape(x,(-1,self.num_T,self.num_H,self.d_k)) # (B,T,D)->(B,T,H,D/H)
        x = tf.transpose(x,perm=[0,2,1,3])  # (B,H,T,D/H)
        x = tf.reshape(x,(-1,self.num_H,self.num_T*self.d_k)) # (B,H,T*D/H)
        return x 

class PerTokenFFN(Layer):
    def __init__(self, num_T, num_D, expansion_ratio=4, **kwargs):
        super().__init__(**kwargs)
        
        # 每个expert的FFN
        self.experts = []
        for i in range(num_T):
            self.experts.append([
                Dense(num_D * expansion_ratio, name=f'expert_{i}_fc1'),
                Activation('gelu'),
                Dense(num_D, name=f'expert_{i}_fc2')
            ])
    
    def call(self, x):        
        outputs=[]
        for i, expert_layers in enumerate(self.experts):
            h = x[:, i, :]
            for layer in expert_layers:
                h = layer(h)
            outputs.append(h)
        
        return tf.stack(outputs, axis=1)
    
class RankMixerLayer(Layer):
    def __init__(self,num_T,num_D,num_H,expansion_ratio,**kwargs):
        super().__init__(**kwargs)
        self.token_mixer = TokenMixer(num_T,num_D,num_H)
        self.per_token_ffn = PerTokenFFN(num_T,num_D,expansion_ratio)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
    def call(self,x):
        print(x.shape)
        mixed_x = self.token_mixer(x)
        print(mixed_x.shape)
        x = self.norm1(x+mixed_x)
        x = self.norm2(x+self.per_token_ffn(x))
        return x

class RankMixer(Layer):
    def __init__(self,num_layers,num_T,num_D,num_H,expansion_ratio,**kwargs):
        super().__init__(**kwargs)
        self.semantic_tokenization = SemanticTokenization(num_T,num_D)
        self.layers_list = []
        for _ in range(num_layers):
            self.layers_list.append(RankMixerLayer(num_T,num_D,num_H,expansion_ratio))
    def call(self,x):
        x = self.semantic_tokenization(x)
        for layer in self.layers_list:
            x = layer(x)
        return tf.reduce_mean(x,axis=1)
    
    
if __name__ == '__main__':
    inputs = Input(shape=(512,))
    rankmixer = RankMixer(num_layers=2,num_T=8,num_D=512,num_H=8,expansion_ratio=4) # to
    outputs = rankmixer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    