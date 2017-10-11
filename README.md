## Deep Crisp Boundaries

Created by Yupei Wang

### Introduction:


Edge detection has made signiﬁcant progress with the help of deep Convolutional Networks (ConvNet). ConvNet based edge detectors approached humanl evel performance on standard benchmarks. We provide a systematical study of these detector outputs, and show that they failed to accurately localize edges, which can be adversarial for tasks that require crisp edge inputs. In addition, we propose a novel reﬁnement architecture to address the challenging problem of learning a crisp edge detector using ConvNet. Our method leverages a top-down backward reﬁnement pathway, and progressively increases the resolution of feature maps to generate crisp edges. Our results achieve promising performance on BSDS500, surpassing human accuracy when using standard criteria, and largely outperforming state-of-the-art methods when using more strict criteria. We further demonstrate the beneﬁt of crisp edge maps for estimating optical ﬂow and generating object proposals.


### Citations

If you are using the code/model provided here in a publication, please cite our paper:

    @InProceedings{xie15hed,
      author = {"Yupei Wang, Xin Zhao and Kaiqi Huang"},
      Title = {Deep Crisp Boundaries},
      Booktitle = "Proceedings of IEEE Conference on Computer Vision and Pattern Recognition",
      Year  = {2017},
    }

### Evaluation results
'examples/CED/resultsImgs_CED': edge maps before NMS with single-scale CED
'examples/CED/resultsImgs_CED_multi': edge maps before NMS with CED
'examples/CED/edge_resultsImgs_CED_multi': edge maps after NMS with CED
'examples/CED/edge_resultsImgs_CED_multi-eval': evaluation results with CED at standard maximal permissible distance d.
'examples/CED/edge_resultsImgs_CED_multi-375-eval': evaluation results with CED at d/2.
'examples/CED/edge_resultsImgs_CED_multi-1875-eval': evaluation results with CED at d/4.

'edge_resultsImgs_CED_multi_VOC_aug': edge maps after NMS with CED_VOC_aug
'edge_resultsImgs_CED_multi_VOC_aug-eval': evaluation results with CED_VOC_aug

  
### Pretrained model

 The pretrained model 'examples/CED/CED.caffemodel' gives ODS=.803 result on BSDS benchmark dataset. And after augmenting the training set with PASCAL Context dataset, the pretrained model 'examples/CED/CED_VOC_aug.caffemodel' gives better resutl ODS=.815

### Installing 
 0. Install prerequisites for Caffe(http://caffe.berkeleyvision.org/installation.html#prequequisites)
 0. Install caffe

### Training HED
To reproduce our results on BSDS500 dataset:
 0. data: Download the datasets
 0. initial model: Use pretrained HED model at 'examples/hed/hed_vgg16.caffemodel'
 0. run the python script **python solve.py** in examples/hed

### Testing HED
Please refer to the original HED(https://github.com/s9xie/hed).
 
For NMS, we used Piotr's Structured Forest matlab toolbox(https://github.com/pdollar/edges). 


### Acknowledgment: 
This code is based on HED. Thanks to the contributors of HED.

@inproceedings{xie2015holistically,
  title={Holistically-nested edge detection},
  author={Xie, Saining and Tu, Zhuowen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1395--1403},
  year={2015}
}


# CED
