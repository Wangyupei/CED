// ------------------------------------------------------------------
// Periodic shuffle operation
// Copyright (c) 2016 Georgia Tech
// Licensed under The MIT License 
// Written by Yi Li
// ------------------------------------------------------------------

#ifndef CAFFE_PERIODIC_SHUFFLE_LAYER_HPP_
#define CAFFE_PERIODIC_SHUFFLE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Shuffle the input blob by changing the memory order of the data.
 *        (N, C, H, W) -> (N, C/n^2, nH, nW) 
 *        This is used for sub-pixel convolution
 * TODO: thorough documentation for Forward, Backward, and proto params.
 */


template <typename Dtype>
class PeriodicShuffleLayer : public Layer<Dtype> {
 public:
  explicit PeriodicShuffleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PeriodicShuffle"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_output_;
  int group_size_;

  int channels_;
  int height_;
  int width_;

  int shuffled_channels_;
  int shuffled_height_;
  int shuffled_width_;
};

}  // namespace caffe

#endif  // CAFFE_PERIODIC_SHUFFLE_LAYER_HPP_