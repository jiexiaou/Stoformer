# Stochastic Window Transformer for Image Restoration (NeurIPS 2022)
 <b>Jie Xiao, <a href='https://xueyangfu.github.io'>Xueyang Fu</a>, Zheng-Jun Zha, Feng Wu</b>

> **Abstract:** *Thanks to the strong representation ability, transformers have attained impressive results for image restoration. However, existing transformers do not carefully take into account the particularities of image restoration. Basically, image restoration requires that the ideal approach should be invariant to translation of degradation, i.e., undesirable degradation should be removed irrespective of its position within the image. Moreover, local relationships play a vital role and should be faithfully exploited for recovering clean images. Nevertheless, most of transformers have resorted to either fixed local window based or global attention, which unfortunately breaks the translation invariance and further causes huge loss of local relationships. To address these issues, we propose an elegant stochastic window strategy for transformers. Specifically, we introduce the window partition with stochastic shift to replace the original fixed window partition for training and elaborate the layer expectation propagation algorithm to efficiently approximate the expectation of the induced stochastic transformer for testing. The stochastic window transformer can not only enjoy powerful representation but also maintain the desired property of translation invariance and locality. Experiments validate the stochastic window strategy consistently improves performance on various image restoration tasks (image deraining, denosing, and deblurring) by significant margins.*
## Method
![Stoformer](figs/method.png)
## The code will be available.
## Contact
Please contact us if there is any question(ustchbxj@mail.ustc.edu.cn).