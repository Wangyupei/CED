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
__global__ void PeriodicShuffleFwdKernel(const int nthreads,
  const Dtype* bottom_data, 
  const int width, const int height, const int channels, const int group_size, 
  const int shuffled_width, const int shuffled_height, const int shuffled_channels,
  Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n1=n, c1, h1, w1
    int n = index / width / height / channels;
    int c1 = (index / width / height) % channels;
    int h1 = (index / width) % height;
    int w1 = index % width;
    // -> n2=n, c2, h2, w2
    int c2 = c1 % shuffled_channels;
    int h2 = group_size * h1 + (c1 / shuffled_channels) / group_size;
    int w2 = group_size * w1 + (c1 / shuffled_channels) % group_size;
      // -> idx
    int idx = n * channels * width * height + 
              c2 * shuffled_width * shuffled_height + h2 * shuffled_width + w2;
    top_data[idx] = bottom_data[index];
  }
}

template <typename Dtype>
__global__ void PeriodicShuffleBwdKernel(const int nthreads,
  const Dtype* top_diff, 
  const int width, const int height, const int channels, const int group_size, 
  const int shuffled_width, const int shuffled_height, const int shuffled_channels,
  Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n1=n, c1, h1, w1
    int n = index / width / height / channels;
    int c1 = (index / width / height) % channels;
    int h1 = (index / width) % height;
    int w1 = index % width;
    // -> n2=n, c2, h2, w2
    int c2 = c1 % shuffled_channels;
    int h2 = group_size * h1 + (c1 / shuffled_channels) / group_size;
    int w2 = group_size * w1 + (c1 / shuffled_channels) % group_size;
      // -> idx
    int idx = n * channels * width * height + 
              c2 * shuffled_width * shuffled_height + h2 * shuffled_width + w2;
    bottom_diff[index] = top_diff[idx];
  }
}
  
template <typename Dtype>
void PeriodicShuffleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  PeriodicShuffleFwdKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, width_, height_, channels_, group_size_,
      shuffled_width_, shuffled_height_, shuffled_channels_, top_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void PeriodicShuffleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  PeriodicShuffleBwdKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, width_, height_, channels_, group_size_,
      shuffled_width_, shuffled_height_, shuffled_channels_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(PeriodicShuffleLayer);

}  // namespace caffe