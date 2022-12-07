# Stochastic Window Transformer for Image Restoration (NeurIPS 2022--Spotlight)
 <b>Jie Xiao, <a href='https://xueyangfu.github.io'>Xueyang Fu</a>, Feng Wu, Zheng-Jun Zha</b>

Paper: [Link](https://openreview.net/pdf/d68b5cc870f601857a724512a93510fa14a372c8.pdf)
> **Abstract:** *Thanks to the strong representation ability, transformers have attained impressive results for image restoration. However, existing transformers do not carefully take into account the particularities of image restoration. Basically, image restoration requires that the ideal approach should be invariant to translation of degradation, i.e., undesirable degradation should be removed irrespective of its position within the image. Moreover, local relationships play a vital role and should be faithfully exploited for recovering clean images. Nevertheless, most of transformers have resorted to either fixed local window based or global attention, which unfortunately breaks the translation invariance and further causes huge loss of local relationships. To address these issues, we propose an elegant stochastic window strategy for transformers. Specifically, we introduce the window partition with stochastic shift to replace the original fixed window partition for training and elaborate the layer expectation propagation algorithm to efficiently approximate the expectation of the induced stochastic transformer for testing. The stochastic window transformer can not only enjoy powerful representation but also maintain the desired property of translation invariance and locality. Experiments validate the stochastic window strategy consistently improves performance on various image restoration tasks (image deraining, denosing, and deblurring) by significant margins.*
## Method
![Stoformer](figs/method.png)
## Data Preparation
- Deblur: [GoPro](https://github.com/swz30/Restormer/tree/main/Motion_Deblurring)
## Train
```
python train_Deblur.py --arch Stoformer --save_dir save_path --train_dir GoPro/train --val_dir GoPro/val --nepoch 600 --embed 32 --checkpoint 100 --optimizer adam --lr_initial 3e-4 --train_workers 4 --env _GoPro --gpu '0,1' --train_ps 256 --batch_size 8 --use_grad_clip
python train_Deblur.py --arch Fixformer --save_dir save_path --train_dir GoPro/train --val_dir GoPro/val --nepoch 600 --embed 32 --checkpoint 100 --optimizer adam --lr_initial 3e-4 --train_workers 4 --env _GoPro --gpu '0,1' --train_ps 256 --batch_size 8 --use_grad_clip
```
## Test
```
python test_Deblur.py --arch Stoformer --gpu '0,1' --input_dir GoPro/test/input --embed_dim 32 --result_dir result/GoPro++ --weights sto_model_path --batch_size 8 --crop_size 512 --overlap_size 32
python test_Deblur.py --arch Stoformer --gpu '0,1' --input_dir GoPro/test/input --embed_dim 32 --result_dir result/GoPro-+ --weights fix_model_path --batch_size 8 --crop_size 512 --overlap_size 32
python test_Deblur.py --arch Fixformer --gpu '0,1' --input_dir GoPro/test/input --embed_dim 32 --result_dir result/GoPro+- --weights sto_model_path --batch_size 8 --crop_size 512 --overlap_size 32
python test_Deblur.py --arch Fixformer --gpu '0,1' --input_dir GoPro/test/input --embed_dim 32 --result_dir result/GoPro-- --weights fix_model_path --batch_size 8 --crop_size 512 --overlap_size 32
```
## Pretrained Model
- Deblur: <a href="https://drive.google.com/file/d/1pURNZs24nXQqEzOJvFtC1wrNMasOQ0lQ/view?usp=share_link">stochastic_model</a> and <a href="https://drive.google.com/file/d/1hObIreDcYJejx9RmYrHOt5feLf9QM7BH/view?usp=share_link">fix_model</a>
## Evaluation
- Deblur: <a href="evaluategopro.m">evaluategopro.m</a>
## Acknowledgement
We refer to [Uformer](https://github.com/ZhendongWang6/Uformer) and [Restormer](https://github.com/swz30/Restormer). Thanks for sharing.
## Citation
```
@inproceedings{xiao2022stochastic,
  title={Stochastic Window Transformer for Image Restoration},
  author={Xiao, Jie and Fu, Xueyang and Wu, Feng and Zha, Zheng-Jun},
  booktitle={NeurIPS},
  year={2022}}
```
## Contact
Please contact us if there is any question(ustchbxj@mail.ustc.edu.cn).