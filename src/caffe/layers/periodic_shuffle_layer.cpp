// ------------------------------------------------------------------
// Periodic shuffle operation
// Copyright (c) 2016 Georgia Tech
// Licensed under The MIT License 
// Written by Yi Li
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/periodic_shuffle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void PeriodicShuffleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PeriodicShuffleParameter periodic_shuffle_param = this->layer_param_.periodic_shuffle_param();
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  int num_axes = bottom[0]->num_axes();
  CHECK_EQ(num_axes, 4)
      << "Only support 4D blob.";
  // get params and check them
  num_output_ = periodic_shuffle_param.num_output();
  group_size_ = periodic_shuffle_param.group_size();
  CHECK_GT(num_output_, 0)
      << "num_output must be > 0";
  CHECK_GT(group_size_, 0)
      << "group_size must be > 0";
}

template <typename Dtype>
void PeriodicShuffleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // get C, H, W from bottom
  channels_ = bottom[0]->shape(1);
  CHECK_EQ(channels_, num_output_*group_size_*group_size_)
      << "Input number of channels does not match ouput channels";
  height_ = bottom[0]->shape(2);
  width_ = bottom[0]->shape(3);
  // calculate C, H, W for top
  shuffled_channels_ = num_output_;
  shuffled_height_ = height_ * group_size_;
  shuffled_width_ = width_ * group_size_;
  // Reshape top blob
  top[0]->Reshape(bottom[0]->shape(0), shuffled_channels_, shuffled_height_,
      shuffled_width_);
}

template <typename Dtype>
void PeriodicShuffleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int count = bottom[0]->count();
  // permute (n1, c1, h1, w1) -> (n2, c2, h2, w2)
  int c1, h1, w1;
  int c2, h2, w2;
  int n, idx;
  for (int i = 0; i < count; i++) {
    // n1=n, c1, h1, w1
    n = i / width_ / height_ / channels_;
    c1 = (i / width_ / height_) % channels_;
    h1 = (i / width_) % height_;
    w1 = i % width_;
    // -> n2=n, c2, h2, w2
    c2 = c1 % shuffled_channels_;
    h2 = group_size_ * h1 + (c1 / shuffled_channels_) / group_size_;
    w2 = group_size_ * w1 + (c1 / shuffled_channels_) % group_size_;
    // -> idx
    idx = n * channels_* width_* height_ + 
          c2 * shuffled_width_ * shuffled_height_ + h2 * shuffled_width_ + w2;
    top_data[idx] = bottom_data[i];
  }
}

template <typename Dtype>
void PeriodicShuffleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  // permute (n1, c1, h1, w1) -> (n2, c2, h2, w2)
  int c1, h1, w1;
  int c2, h2, w2;
  int n, idx;
  for (int i = 0; i < count; i++) {
    // n1=n, c1, h1, w1
    n = i / width_ / height_ / channels_;
    c1 = (i / width_ / height_) % channels_;
    h1 = (i / width_) % height_;
    w1 = i % width_;
    // -> n2=n, c2, h2, w2
    c2 = c1 % shuffled_channels_;
    h2 = group_size_ * h1 + (c1 / shuffled_channels_) / group_size_;
    w2 = group_size_ * w1 + (c1 / shuffled_channels_) % group_size_;
    // -> idx
    idx = n * channels_* width_* height_ + 
          c2 * shuffled_width_ * shuffled_height_ + h2 * shuffled_width_ + w2;
    bottom_diff[i] = top_diff[idx];
  }
}

#ifdef CPU_ONLY
STUB_GPU(PeriodicShuffleLayer);
#endif

INSTANTIATE_CLASS(PeriodicShuffleLayer);
REGISTER_LAYER_CLASS(PeriodicShuffle);

}  // namespace caffe