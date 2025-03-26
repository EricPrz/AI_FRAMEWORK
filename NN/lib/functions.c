#include <stdio.h>
#include <stdlib.h>

float *getMaxPool2D(int dim0, int dim1, int dim2, int dim3, float *array,
                    int k_size, int in_channels, int stride, int dilation,
                    int padding, int *extracted_from_pos) {
  // Compute output spatial size
  int n = (dim2 + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;

  // Allocate memory for output array
  float *pooledArray =
      (float *)malloc(dim0 * in_channels * n * n * sizeof(float));
  if (pooledArray == NULL) {
    return NULL; // Memory allocation failed
  }

  // Compute max pooling
  for (int N = 0; N < dim0; N++) {
    for (int ch = 0; ch < in_channels; ch++) {
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
          // Get starting position
          int base_y = j * stride;
          int base_x = i * stride;

          // Ensure the first element is within bounds
          int maxIdx = N * dim1 * dim2 * dim3 + ch * dim2 * dim3 +
                       base_y * dim3 + base_x;
          float max = array[maxIdx]; // A very small number (initialize safely)

          for (int y = 0; y < k_size; y++) {
            for (int x = 0; x < k_size; x++) {
              int idx_y = base_y + y * dilation;
              int idx_x = base_x + x * dilation;

              // Bounds check to avoid invalid memory access
              if (idx_y < dim2 && idx_x < dim3) {
                int pos = N * dim1 * dim2 * dim3 + ch * dim2 * dim3 +
                          idx_y * dim3 + idx_x;
                float selected = array[pos];

                if (selected > max) {
                  max = selected;
                  maxIdx = pos;
                }
              }
            }
          }

          // Store the max value in the flat array
          pooledArray[N * in_channels * n * n + ch * n * n + j * n + i] = max;
          extracted_from_pos[N * in_channels * n * n + ch * n * n + j * n + i] =
              maxIdx;
        }
      }
    }
  }

  return pooledArray;
}

void setSects(float *array, float *pooled, int *posarr, int N, int chs, int zoneNumbers, int dim, int stride, int k_size, int dilation) {
  for (int n = 0; n < N; n++) {
    for (int ch = 0; ch < chs; ch++){
      for (int j = 0; j < zoneNumbers; j++) {
        for (int i = 0; i < zoneNumbers; i++) {
            
          // Get starting position
          int base_y = j * stride;
          int base_x = i * stride;
        
        
          for (int y = 0; y < k_size; y++) {
            for (int x = 0; x < k_size; x++) {
              int idx_y = base_y + y * dilation;
              int idx_x = base_x + x * dilation;

              int pos = n * chs * dim * dim 
                        + ch * dim * dim 
                        + idx_y * dim 
                        + idx_x;

              int pooledPos = n * chs * zoneNumbers * zoneNumbers * k_size * k_size
                              + ch * zoneNumbers * zoneNumbers * k_size * k_size
                              + j * zoneNumbers * k_size * k_size
                              + i * k_size * k_size
                              + y * k_size
                              + x;
              
              pooled[pooledPos] = array[pos];

              posarr[pooledPos] = pos;
            }
          }
         }
      }
    }
  }  
}

float *getSects(float *array, int *posarr, int N, int chs, int zoneNumbers, int dim, int stride, int k_size, int dilation) {
  // Allocate memory for pooled
  float *pooled = (float *)malloc(N * chs * zoneNumbers * zoneNumbers * k_size * k_size * sizeof(float));
  
  // Precompute some values to avoid recalculating in each loop
  int chs_dim_dim = chs * dim * dim;
  int ch_zone_size = zoneNumbers * zoneNumbers * k_size * k_size;
  int zone_size = zoneNumbers * k_size * k_size;

  for (int n = 0; n < N; n++) {
    for (int ch = 0; ch < chs; ch++) {
      for (int j = 0; j < zoneNumbers; j++) {
        for (int i = 0; i < zoneNumbers; i++) {
          // Get starting position
          int base_y = j * stride;
          int base_x = i * stride;

          for (int y = 0; y < k_size; y++) {
            for (int x = 0; x < k_size; x++) {
              int idx_y = base_y + y * dilation;
              int idx_x = base_x + x * dilation;

              int pos = (n * chs_dim_dim) + (ch * dim * dim) + (idx_y * dim) + idx_x;

              // Precompute pooledPos to avoid redundant calculation
              int pooledPos = (n * chs * ch_zone_size) 
                               + (ch * ch_zone_size) 
                               + (j * zone_size) 
                               + (i * k_size * k_size) 
                               + (y * k_size) 
                               + x;
              
              // Use pointer arithmetic for faster memory access
              pooled[pooledPos] = array[pos];
              posarr[pooledPos] = pos;
            }
          }
        }
      }
    }
  }

  return pooled;
}


void cleanPtr(float *ptr) {
  if (ptr != NULL) {
    free(ptr);  // Free the allocated memory
    ptr = NULL; // Avoid dangling pointer issues
  }
}
