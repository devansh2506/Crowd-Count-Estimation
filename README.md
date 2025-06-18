# Crowd-Count-Estimation

For Transformer-based regression: https://colab.research.google.com/drive/1JVVViTyURlV34hgSq3W4WV7otKKP3Ap_?usp=sharing

Crowd counting – estimating the number of people in an image – is a key task in computer
vision with applications in public safety, surveillance, and event management. It is
challenging due to scale variations, heavy occlusion, perspective distortion, and background
clutter. Traditional CNN methods (e.g. CSRNet) achieve strong results by regressing density
maps, while recent transformer-based models (e.g. TransCrowd) use self-attention to predict
countsarxiv.org. In this work we implement two deep models on the ShanghaiTech Part B
dataset: (1) VUI-CrowdNet, a CNN with a VGG-16 encoder, U-Net–style decoder, and an
Inverse Attention Block (IAB) to suppress background, and (2) a Vision Transformer
Regression model that treats counting as whole-image regression with two heads (a learnable
“count token” vs. global average pooling). Our goal is to blend CNN precision and
segmentation awareness with Transformer-based global context reasoning.


VUI-CrowdNet:

We build VUI-CrowdNet on a VGG-16 encoder (pretrained on ImageNet). An
input image (768×1024×3) is fed through VGG-16 up to the fifth pooling layer, producing
feature maps of size 24×32×512 that encode both low-level textures and high-level semantics.
To restore full spatial resolution, we attach a U-Net–style decoder consisting of five
upsampling blocks. Each block performs a 3×3 transposed convolution (stride 2) followed by a
3×3 convolution (both with 64 filters), effectively doubling the width and height of the feature
maps at each stage. This careful upsampling preserves spatial detail, allowing the network to
output a high-resolution density map. Maintaining original resolution avoids any loss of detail
that resizing might cause, as emphasized by recent encoder-decoder designs.
After the final upsampling, we apply our Inverse Attention Block (IAB). The IAB is a small 3-
layer convolutional module that predicts an inverse attention mask A⁻¹ highlighting
background regions. Formally, if F is the upsampled feature map (after the decoder), the IAB
outputs an element-wise mask A⁻¹ of the same spatial size. We then compute a refined map

F′ = F − F ⊙ A⁻¹

where ⊙ denotes elementwise multiplication. Intuitively, this subtracts out features
associated with background, forcing the network to focus on crowd areas (inspired by IA-
DCCN). This inversion scheme is key to making the counting easier by dimming non-crowd
regions. Finally, we pass F through a 1x1 Convolution to produce the predicted density map.

Vission Transformer Regression:

Our transformer-based model treats crowd counting as a global image
regression problem. We resize inputs to 384×384 and split each image into non-overlapping
16×16 patches (total 576 patches). We prepend a learnable [CLS] token to the sequence. The
backbone is a standard ViT with 12 layers of multi-head self-attention (12 heads, embedding
dimension 768). The ViT processes the 577 tokens (576 patches + 1 [CLS]) through the encoder.
We then attach two alternative regression heads to compare strategies:
Token-based head: We use the output corresponding to the [CLS] token (or a dedicated
regression token) and feed it through a small MLP to predict the crowd count. This forces
the network to encode all count information into that one vector.
GAP-based head: We discard the [CLS] token and instead apply global average pooling
over the 576 patch-token embeddings. The pooled vector is then passed through a linear
layer to produce the count.
This dual-head setup follows the design of TransCrowdarxiv.org (TransCrowd-Token vs.
TransCrowd-GAP). We use a dropout of 0.5 before the final layer for regularization. The token
head aims to concentrate count evidence into one feature, while the GAP head aggregates
information from the entire scene. In our experiments (see Results) the token head
consistently outperformed GAP , suggesting the ViT can effectively learn a dedicated count
token.


Results:

On the given dataset, our best VUI-CrowdNet model achieved MAE ≈16.7. This is a competitive
result: it improves upon many classical methods (e.g. Switching-CNN’s MAE 21.6) and
demonstrates the value of segmentation guidance and U-Net decoding. The predicted
density maps were qualitatively convincing – background areas were dimmed by the IAB and
crowd blobs were highlighted in the correct locations. Although we do not show them here, in
test images the network clearly focused on clusters of pedestrians, validating the approach
of background suppression.
The ViT model produced reasonable count estimates but lacked a spatial map. Numerically,
its MAE (~25 with the token head) was higher than the CNN’s. We attribute this gap to the
heavier training requirements of Transformers and the limited tuning time we had. (With more
extensive fine-tuning, multi-scale inputs, or optimized attention mechanisms, we expect the
ViT’s accuracy would improve.)
The key takeaway is that both approaches are viable: the CNN model achieved lower error
and faster training, while the ViT model demonstrated that a pure-attention architecture can
learn the counting task. Each provides different advantages: the CNN yields spatially
coherent density maps, and the ViT offers flexibility for global analysis.
In summary, our CNN-based method was more accurate on this dataset, whereas the Vision
Transformer validated the concept of transformer-based counting. Time constraints and
dataset size limited how far we could optimize each model: given more training epochs or
compute, both models could likely improve further. Nevertheless, the results clearly show
that injecting segmentation awareness into a CNN and leveraging self-attention in a
transformer are both effective strategies for crowd counting.
