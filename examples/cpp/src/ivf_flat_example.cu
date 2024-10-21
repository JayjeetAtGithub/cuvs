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

#include "CLI11.hpp"
#include "common.cuh"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/stats/neighborhood_recall.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/prefetch_resource_adaptor.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <chrono>
#include <cstdint>
#include <optional>

namespace {
/// MR factory functions
inline auto make_cuda() {
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

inline auto make_async() {
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

inline auto make_managed() {
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

inline auto make_prefetch() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      make_managed());
}

inline auto make_managed_pool() {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      make_managed(), rmm::percent_of_free_device_memory(50));
}

inline auto make_prefetch_pool() {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      make_prefetch(), rmm::percent_of_free_device_memory(50));
}

inline std::shared_ptr<rmm::mr::device_memory_resource>
create_memory_resource(std::string const &allocation_mode) {
  if (allocation_mode == "cuda")
    return make_cuda();
  if (allocation_mode == "async")
    return make_async();
  if (allocation_mode == "managed")
    return make_managed();
  if (allocation_mode == "prefetch")
    return make_prefetch();
  if (allocation_mode == "managed_pool")
    return make_managed_pool();
  if (allocation_mode == "prefetch_pool")
    return make_prefetch_pool();
  return make_managed();
}
} // namespace

void ivf_search(raft::device_resources const &res,
                raft::device_matrix_view<const float, int64_t> dataset,
                raft::device_matrix_view<const float, int64_t> queries,
                int64_t n_list, int64_t n_probe, int64_t top_k) {
  using namespace cuvs::neighbors;
  std::cout << "Performing IVF-FLAT search" << std::endl;

  // Build the IVF-FLAT index
  ivf_flat::index_params index_params;
  index_params.n_lists = n_list;
  index_params.kmeans_trainset_fraction = 0.1;
  index_params.kmeans_n_iters = 100;
  index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  auto s = std::chrono::high_resolution_clock::now();
  auto index = ivf_flat::build(res, index_params, dataset);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Train and index: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;
  std::cout << "[INFO] Number of clusters " << index.n_lists()
            << ", number of vectors added to index " << index.size()
            << std::endl;

  // Define arrays to hold search output results
  int64_t n_queries = queries.extent(0);
  auto neighbors =
      raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
  auto distances =
      raft::make_device_matrix<float, int64_t>(res, n_queries, top_k);

  // Perform the search operation
  ivf_flat::search_params search_params;
  search_params.n_probes = n_probe;
  s = std::chrono::high_resolution_clock::now();
  ivf_flat::search(res, search_params, index, queries, neighbors.view(),
                   distances.view());
  raft::resource::sync_stream(res);
  e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Search: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // Brute force search for reference
  auto reference_neighbors =
      raft::make_device_matrix<int64_t, int64_t>(res, n_queries, top_k);
  auto reference_distances =
      raft::make_device_matrix<float, int64_t>(res, n_queries, top_k);
  auto brute_force_index = cuvs::neighbors::brute_force::build(res, dataset);
  cuvs::neighbors::brute_force::search(res, brute_force_index, queries,
                                       reference_neighbors.view(),
                                       reference_distances.view());
  raft::resource::sync_stream(res);
  float const recall_scalar = 0.0;
  auto recall_value = raft::make_host_scalar(recall_scalar);
  raft::stats::neighborhood_recall(
      res, raft::make_const_mdspan(neighbors.view()),
      raft::make_const_mdspan(reference_neighbors.view()), recall_value.view());
  std::cout << "[INFO] Recall@" << top_k << ": " << recall_value(0)
            << std::endl;
}

int main(int argc, char **argv) {
  CLI::App app{"Run CUVS Benchmarks"};
  argv = app.ensure_utf8(argv);

  std::string dataset_dir;
  app.add_option("-d,--dataset-dir", dataset_dir, "Path to the dataset");

  std::string algo = "ivf";
  app.add_option("--algo", algo, "Algorithm to run: cagra or ivf");

  std::string mem_type = "cuda";
  app.add_option("--mem-type", mem_type,
                 "Memory type: cuda / async / managed / prefetch");

  int64_t cuda_device = 0;
  app.add_option("--cuda-device", cuda_device, "The CUDA device to use");

  int64_t learn_limit = 10000;
  app.add_option("--learn-limit", learn_limit,
                 "Limit the number of learn vectors");

  int64_t search_limit = 10000;
  app.add_option("--search-limit", search_limit,
                 "Limit the number of search vectors");

  int64_t top_k = 10;
  app.add_option("-k,--top-k", top_k, "Number of nearest neighbors");

  int64_t n_probe = 32;
  app.add_option("--n-probe", n_probe, "Number of probes");

  CLI11_PARSE(app, argc, argv);

  if (dataset_dir.empty()) {
    std::cerr << "[ERROR] Please provide a dataset directory" << std::endl;
    return 1;
  }

  // Set the memory resources
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  auto mr = create_memory_resource(mem_type);
  auto stats_mr =
      rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(
          mr.get());
  rmm::mr::set_current_device_resource(&stats_mr);

  // Read the dataset files
  std::string dataset_path = dataset_dir + "/dataset.bin";
  std::string query_path = dataset_dir + "/query.bin";

  auto dataset_device =
      read_bin_dataset<float, int64_t>(res, dataset_path.c_str(), learn_limit);
  auto queries_device =
      read_bin_dataset<float, int64_t>(res, query_path.c_str(), 10'000);

  int64_t n_dataset = dataset_device.extent(0);
  int64_t d_dataset = dataset_device.extent(1);
  int64_t n_queries = queries_device.extent(0);
  int64_t d_queries = queries_device.extent(1);

  std::cout << "[INFO] Dataset shape: " << n_dataset << " x " << d_dataset
            << std::endl;
  std::cout << "[INFO] Queries shape: " << n_queries << " x " << d_queries
            << std::endl;

  // Set the index and search params
  int64_t n_list = int64_t(4 * std::sqrt(n_dataset));

  if (algo == "ivf") {
    ivf_search(res, raft::make_const_mdspan(dataset_device.view()),
               raft::make_const_mdspan(queries_device.view()), n_list, n_probe,
               top_k);
  }

  raft::resource::sync_stream(res);
  std::cout << "[INFO] Peak memory usage: "
            << (stats_mr.get_bytes_counter().peak / 1048576.0) << " MiB\n";
}
