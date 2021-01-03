# Transformer_chatbot
build transformer model to be a chatbot with attention.

![image](https://miro.medium.com/max/814/1*rwyRXtMepEk0nClIlA_F3w.png)

## Attention
### Scaled dot product Attention

![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYLZ9oH23RUNS8LtHPkGrPdmG8VM_OJHmMjg&usqp=CAU)

The scaled dot-product attention function used by the transformer takes three inputs: Q (query), K (key), V (value). The equation used to calculate the attention weights is:

![image](https://glassboxmedicine.files.wordpress.com/2019/08/attention-equation.png?w=616)

As the softmax normalization is done on the key, its values decide the amount of importance given to the query.

The output represents the multiplication of the attention weights and the value vector. This ensures that the words we want to focus on are kept as is and the irrelevant words are flushed out.

The dot-product attention is scaled by a factor of square root of the depth. This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax function where it has small gradients resulting in a very hard softmax.

For example, consider that query and key have a mean of 0 and variance of 1. Their matrix multiplication will have a mean of 0 and variance of dk. Hence, square root of dk is used for scaling (and not any other number) because the matmul of query and key should have a mean of 0 and variance of 1, so that we get a gentler softmax.

The mask is multiplied with -1e9 (close to negative infinity). This is done because the mask is summed with the scaled matrix multiplication of query and key and is applied immediately before a softmax. The goal is to zero out these cells, and large negative inputs to softmax are near zero in the output.

### Multi-head attention

![image](https://paperswithcode.com/media/methods/multi-head-attention_l1A3G7a.png)

Multi-head attention consists of four parts:

Linear layers and split into heads.
Scaled dot-product attention.
Concatenation of heads.
Final linear layer.
Each multi-head attention block gets three inputs; Q (query), K (key), V (value). These are put through linear (Dense) layers and split up into multiple heads.

The scaled_dot_product_attention defined above is applied to each head (broadcasted for efficiency). An appropriate mask must be used in the attention step. The attention output for each head is then concatenated (using tf.transpose, and tf.reshape) and put through a final Dense layer.

Instead of one single attention head, query, key, and value are split into multiple heads because it allows the model to jointly attend to information at different positions from different representational spaces. After the split each head has a reduced dimensionality, so the total computation cost is the same as a single head attention with full dimensionality.

### Positional encoding
Since this model doesn't contain any recurrence or convolution, positional encoding is added to give the model some information about the relative position of the words in the sentence.

The positional encoding vector is added to the embedding vector. Embeddings represent a token in a d-dimensional space where tokens with similar meaning will be closer to each other. But the embeddings do not encode the relative position of words in a sentence. So after adding the positional encoding, words will be closer to each other based on the similarity of their meaning and their position in the sentence, in the d-dimensional space.

The formula for calculating the positional encoding is as follows:

![image](https://miro.medium.com/max/2174/0*PuV87IVkFin98EZW.png)
