## Deep Crisp Boundaries

### Introduction:

Edge detection has made significant progress with the help of deep Convolutional Networks (ConvNet). These ConvNet based edge detectors have approached human level performance on standard benchmarks. We provide a systematical study of these detectors' outputs. We show that the detection results did not accurately localize edge pixels, which can be adversarial for tasks that require crisp edge inputs. As a remedy, we propose a novel refinement architecture to address the challenging problem of learning a crisp edge detector using ConvNet. Our method leverages a top-down backward refinement pathway, and progressively increases the resolution of feature maps to generate crisp edges. Our results achieve superior performance, surpassing human accuracy when using standard criteria on BSDS500, and largely outperforming state-of-the-art methods when using more strict criteria. More importantly, we demonstrate the benefit of crisp edge maps for several important applications in computer vision, including optical flow estimation, object proposal generation and semantic segmentation.


### Citations

If you are using the code/model provided here in a publication, please cite our paper:

    @InProceedings{Wang17CED,
      author = {"Yupei Wang, Xin Zhao and Kaiqi Huang"},
      Title = {Deep Crisp Boundaries},
      Booktitle = "Proceedings of IEEE Conference on Computer Vision and Pattern Recognition",
      Year  = {2017},
    }
    
    @article{Wang18Crisp,
      title={Deep Crisp Boundaries: From Boundaries to Higher-level Tasks},
      author={ Yupei Wang, Xin Zhao, Yin Li and Kaiqi Huang},
      journal={TIP},
      year={2018},
      publisher={IEEE}
}



### Evaluation results can be downloaded at(https://drive.google.com/file/d/1NYvlJYVonLs3twtYPDBV8l0x9ooxIFCY/view?usp=sharing)
'CED_results/resultsImgs_CED': edge maps before NMS with single-scale CED

'CED_results/resultsImgs_CED_multi': edge maps before NMS with CED

'CED_results/edge_resultsImgs_CED_multi': edge maps after NMS with CED

'CED_results/edge_resultsImgs_CED_multi-eval': evaluation results with CED at standard maximal permissible distance d.

'CED_results/edge_resultsImgs_CED_multi-375-eval': evaluation results with CED at d/2.

'CED_results/edge_resultsImgs_CED_multi-1875-eval': evaluation results with CED at d/4.

'CED_results/edge_resultsImgs_CED_multi_VOC_aug': edge maps after NMS with CED_VOC_aug

'CED_results/edge_resultsImgs_CED_multi_VOC_aug-eval': evaluation results with CED_VOC_aug

'CED_results/resultsImgs_Res50_CED_multi_VOC_aug': edge maps before NMS with Res50_CED_multi_VOC_aug

'CED_results/edge_resultsImgs_Res50_CED_multi_VOC_aug': edge maps after NMS with Res50_CED_multi_VOC_aug

'CED_results/edge_resultsImgs_Res50_CED_multi_VOC_aug-eval': evaluation results with Res50_CED_multi_VOC_aug

### Training and testing

Please refer to the original HED(https://github.com/s9xie/hed).
 
For NMS, we used Piotr's Structured Forest matlab toolbox(https://github.com/pdollar/edges). 

Note: Initialize with the pretrained HED(https://drive.google.com/file/d/1R_kUd028OMWnPSRNBJTEkUQDrYOoSP7P/view?usp=sharing)

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
