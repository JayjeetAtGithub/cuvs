/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/stats/neighborhood_recall.cuh>

#include <rmm/aligned.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/prefetch_resource_adaptor.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <optional>
#include <chrono>


/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

inline auto make_prefetch() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(make_managed());
}

inline std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
  std::string const& allocation_mode)
{
  if (allocation_mode == "cuda") return make_cuda();
  if (allocation_mode == "async") return make_async();
  if (allocation_mode == "managed") return make_managed();
  if (allocation_mode == "prefetch") return make_prefetch();
  return make_managed();
}

void ivf_search(raft::device_resources const& res,
                raft::device_matrix_view<const float, int64_t> dataset,
                raft::device_matrix_view<const float, int64_t> queries,
                int64_t n_list,
                int64_t n_probe,
                int64_t top_k)
{
  using namespace cuvs::neighbors;
  std::cout << "Performing IVF-FLAT search" << std::endl;

  // Build the IVF-FLAT index
  ivf_flat::index_params index_params;
  index_params.n_lists                  = n_list;
  index_params.kmeans_trainset_fraction = 0.1;
  index_params.kmeans_n_iters           = 100;
  index_params.metric                   = cuvs::distance::DistanceType::L2Expanded;
  auto s = std::chrono::high_resolution_clock::now();
  auto index = ivf_flat::build(res, index_params, dataset);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
    << "[TIME] Train and index: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
    << " ms" << std::endl;
  std::cout << "[INFO] Number of clusters " << index.n_lists() << ", number of vectors added to index "
            << index.size() << std::endl;

  // Define arrays to hold search output results
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
  auto distances    = raft::make_device_matrix<float, int64_t>(res, n_queries, top_k);

  // Perform the search operation
  ivf_flat::search_params search_params;
  search_params.n_probes = n_probe;
  s = std::chrono::high_resolution_clock::now();
  ivf_flat::search(
    res, search_params, index, queries, neighbors.view(), distances.view());
  e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Search: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // Brute force search for reference
  auto reference_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
  auto reference_distances = raft::make_device_matrix<float, int64_t>(res, n_queries, top_k); 
  auto brute_force_index = cuvs::neighbors::brute_force::build(res, dataset);
  cuvs::neighbors::brute_force::search(res,
                                     brute_force_index,
                                     queries,
                                     reference_neighbors.view(),
                                     reference_distances.view());
  float const recall_scalar = 0.0;
  auto recall_value = raft::make_host_scalar(recall_scalar);
  raft::stats::neighborhood_recall(res,
                                  raft::make_const_mdspan(neighbors.view()),
                                  raft::make_const_mdspan(reference_neighbors.view()),
                                  recall_value.view());
  std::cout << "[INFO] Recall@" << top_k << ": " << recall_value(0) << std::endl;
}

int main(int argc, char **argv)
{
  if (argc != 6) {
    std::cout << argv[0] << " <learn_limit> <n_probe> <algo> <dataset_dir> <mem_type>" << std::endl;
    exit(1);
  }

  // Get params from the user
  int64_t learn_limit = std::stoi(argv[1]);
  int64_t n_probe = std::stoi(argv[2]);
  std::string algo = argv[3];
  std::string dataset_dir = argv[4];
  std::string mem_type = argv[5];

  // Set the memory resources
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  auto mr = create_memory_resource(mem_type);
  auto stats_mr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(mr.get());
  rmm::mr::set_current_device_resource(&stats_mr);

  // Read the dataset files
  std::string dataset_path = dataset_dir + "/dataset.bin";
  std::string query_path = dataset_dir + "/query.bin";
  
  auto dataset_device = read_bin_dataset<float, int64_t>(res, dataset_path.c_str(), learn_limit);
  auto queries_device = read_bin_dataset<float, int64_t>(res, query_path.c_str(), 10'000);

  int64_t n_dataset = dataset_device.extent(0);
  int64_t d_dataset = dataset_device.extent(1);
  int64_t n_queries = queries_device.extent(0);
  int64_t d_queries = queries_device.extent(1);

  std::cout << "Dataset: " << n_dataset << "x" << d_dataset << std::endl;
  std::cout << "Queries: " << n_queries << "x" << d_queries << std::endl;

  // Set the index and search params
  int64_t n_list = int64_t(4 * std::sqrt(n_dataset));
  int64_t top_k = 100;

  if (algo == "ivf") {
    ivf_search(res,
              raft::make_const_mdspan(dataset_device.view()),
              raft::make_const_mdspan(queries_device.view()),
              n_list,
              n_probe,
              top_k);
  }

  res.sync_stream();
  std::cout << "[INFO] Peak memory usage: " << (stats_mr.get_bytes_counter().peak / 1048576.0) << " MB\n";
}
