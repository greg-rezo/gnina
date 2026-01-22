/*
 * torch_model.cpp
 *
 *  Created on: Feb 14, 2024
 *      Author: dkoes
 */

#include "torch_model.h"
#include "common.h"
#include <json/json.h>
#include <string>
#include <chrono>
#include <iomanip>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/Context.h>

using namespace std;
using namespace libmolgrid;

static const string default_recmap(R"(AliphaticCarbonXSHydrophobe 
AliphaticCarbonXSNonHydrophobe 
AromaticCarbonXSHydrophobe 
AromaticCarbonXSNonHydrophobe
Bromine Iodine Chlorine Fluorine
Nitrogen NitrogenXSAcceptor 
NitrogenXSDonor NitrogenXSDonorAcceptor
Oxygen OxygenXSAcceptor 
OxygenXSDonorAcceptor OxygenXSDonor
Sulfur SulfurAcceptor
Phosphorus 
Calcium
Zinc
GenericMetal Boron Manganese Magnesium Iron
)");

static const string default_ligmap(R"(AliphaticCarbonXSHydrophobe 
AliphaticCarbonXSNonHydrophobe 
AromaticCarbonXSHydrophobe 
AromaticCarbonXSNonHydrophobe
Bromine Iodine
Chlorine
Fluorine
Nitrogen NitrogenXSAcceptor 
NitrogenXSDonor NitrogenXSDonorAcceptor
Oxygen OxygenXSAcceptor 
OxygenXSDonorAcceptor OxygenXSDonor
Sulfur SulfurAcceptor
Phosphorus
GenericMetal Boron Manganese Magnesium Zinc Calcium Iron
)");

/** Read in a torch script model from the provided stream */
template <bool isCUDA> TorchModel<isCUDA>::TorchModel(std::istream &in, const string &name, tee *log) {
  c10::Device device = isCUDA ? torch::kCUDA : torch::kCPU;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::ExtraFilesMap extras;
    extras["metadata"] = ""; // only loads if key already present
    module = torch::jit::load(in, device, extras);
    module.to(device);

    string data = extras["metadata"];
    string recmap, ligmap;
    double resolution = 0.5;
    double dimension = 23.5;
    double rscale = 1.0;
    if (data.length() == 0) {
      if (log)
        *log << "WARNING: Torch model missing metadata.  Default grid parameters will be used.\n";
      recmap = default_recmap;
      ligmap = default_ligmap;
    } else {
      Json::Reader reader;
      Json::Value root;
      reader.parse(data, root);

      if (root.isMember("resolution")) {
        resolution = root["resolution"].asDouble();
      } else {
        if (log)
          *log << "WARNING: Resolution not specified in model file.  Using default.\n";
      }
      if (root.isMember("dimension")) {
        dimension = root["dimension"].asDouble();
      } else {
        if (log)
          *log << "WARNING: Dimension not specified in model file.  Using default.\n";
      }
      if (root.isMember("ligmap")) {
        ligmap = root["ligmap"].asString();
      } else {
        if (log)
          *log << "WARNING: ligmap not specified in model file.  Using default.\n";
      }
      if (root.isMember("recmap")) {
        recmap = root["recmap"].asString();
      } else {
        if (log)
          *log << "WARNING: recmap not specified in model file.  Using default.\n";
      }
      if (root.isMember("apply_logistic_loss")) {
        apply_logistic_loss = root["apply_logistic_loss"].asBool();
      }
      if (root.isMember("skip_softmax")) {
        skip_softmax = root["skip_softmax"].asBool();
      }      
      if (root.isMember("radius_scaling")) {
        rscale = root["radius_scaling"].asFloat();
      }
    }

    gmaker.initialize(resolution, dimension, false, rscale);

    // setup typers
    stringstream ligstream(ligmap), recstream(recmap);
    lig_typer = make_shared<FileMappedGninaTyper>(ligstream);
    rec_typer = make_shared<FileMappedGninaTyper>(recstream);

  } catch (const c10::Error &e) {
    throw usage_error("Could not read torch model " + name);
  }
}

static CoordinateSet make_coordset(const vector<float3> &coords, const vector<smt> &smtypes,
                                   shared_ptr<AtomTyper> typer) {
  if (coords.size() != smtypes.size())
    throw internal_error("Shape mismatch", __LINE__);

  vector<float> types;
  types.reserve(coords.size());
  vector<float> radii;
  radii.reserve(coords.size());
  for (unsigned i = 0, n = smtypes.size(); i < n; i++) {
    smt origt = smtypes[i];
    auto t_r = typer->get_int_type(origt);
    int t = t_r.first;
    types.push_back(t);
    radii.push_back(t_r.second);

    if (t < 0 && origt > 1) { // don't warn about hydrogens
      std::cerr << "Unsupported ligand atom type " << GninaIndexTyper::gnina_type_name(origt) << "\n";
    }
  }

  return CoordinateSet(coords, types, radii, typer->num_types());
}

//wrapper to get appropriate grid from an MGrid for template value of isCUDA
template <bool isCUDA>
static Grid<float, 2, isCUDA> get2DGrid(MGrid2f& mgrid);
template<>
Grid<float, 2, true> get2DGrid(MGrid2f& mgrid) { return mgrid.gpu();}
template<>
Grid<float, 2, false> get2DGrid(MGrid2f& mgrid) { return mgrid.cpu();}

template <bool isCUDA>
std::vector<float> TorchModel<isCUDA>::forward(const std::vector<float3> &rec_coords, const std::vector<smt> &rec_types,
                                               const std::vector<float3> &lig_coords, const std::vector<smt> &lig_types,
                                               const vec &center, bool rotate, bool compute_gradient) {

  torch::AutoGradMode enable_grad(compute_gradient);
  // make coordinate sets
  CoordinateSet rec = make_coordset(rec_coords, rec_types, rec_typer);
  CoordinateSet lig = make_coordset(lig_coords, lig_types, lig_typer);

  // set center from ligand if not specified
  float3 gcenter = {center.x(), center.y(), center.z()};
  if (!isfinite(center.x())) {
    gcenter = lig.center();
  }

  CoordinateSet combined(rec, lig);

  Transform transform(gcenter, 0, rotate);
  if (rotate) {
    transform.forward(combined, combined);
  }
  // create grids
  long ntypes = combined.num_types();
  long gd = gmaker.get_first_dim();

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(isCUDA ? torch::kCUDA : torch::kCPU).requires_grad(compute_gradient);
  torch::Tensor gtensor = torch::zeros({1, ntypes, gd, gd, gd}, options);
  Grid<float, 4, isCUDA> out(gtensor.data_ptr<float>(), ntypes, gd, gd, gd);
  gmaker.forward(gcenter, combined, out);

  // evaluate model
  vector<torch::jit::IValue> inputs{gtensor};
  auto result = module.forward(inputs).toTuple()->elements();

  // get results
  auto pose_logit = result[0].toTensor();
  auto pose = skip_softmax ? pose_logit.index({0,1}).item<float>() : torch::softmax(pose_logit, 1).index({0, 1}).item<float>();
  auto affinity = result[1].toTensor()[0].item<float>();

  auto loptions = torch::TensorOptions().dtype(torch::kLong).device(isCUDA ? torch::kCUDA : torch::kCPU);
  torch::Tensor labels = torch::ones({1}, loptions);

  auto loss = apply_logistic_loss ? -torch::log(pose_logit.index({0,1})) : torch::cross_entropy_loss(pose_logit, labels);

  if (compute_gradient) {
    loss.backward(); 
    auto grad = gtensor.grad();
    Grid<float, 4, isCUDA> gridgrad(grad.data_ptr<float>(), ntypes, gd, gd, gd);
    MGrid2f atomic_gradients(combined.size(),3);
    auto coord_grad = get2DGrid<isCUDA>(atomic_gradients);
    gmaker.backward(gcenter,combined, gridgrad, coord_grad);
    if(rotate) {
      transform.backward(coord_grad,coord_grad,false);
    }
    unsigned nr = rec.size();
    unsigned nl = lig.size();
    VINA_CHECK(nr+nl == combined.size());
    gradient_rec.resize(nr);

    auto cg_cpu = atomic_gradients.cpu();
    for(unsigned i = 0; i < nr; i++) {
      gradient_rec[i] = gfloat3(cg_cpu[i][0],cg_cpu[i][1],cg_cpu[i][2]);
    }
    gradient_lig.resize(nl);
    for(unsigned i = 0; i < nl; i++) {
      gradient_lig[i] = gfloat3(cg_cpu[i+nr][0],cg_cpu[i+nr][1],cg_cpu[i+nr][2]);
    }

  }
  vector<float> scores{pose, affinity, loss.item<float>()};
  return scores;
}

template <bool isCUDA> void TorchModel<isCUDA>::getLigandGradient(std::vector<gfloat3> &grad) {
  grad = gradient_lig;
}

template <bool isCUDA> void TorchModel<isCUDA>::getReceptorGradient(std::vector<gfloat3> &grad) {
  grad = gradient_rec;
}

// Multi-ligand batched forward pass
// Uses per-pose voxelization (gmaker.forward) but batches NN inference across all poses
// This is more efficient than calling forward_batch once per ligand because:
// 1. Receptor CoordinateSet is created once and reused
// 2. Single GPU sync at the end instead of per-ligand
// 3. Fewer memory allocations
template <bool isCUDA>
std::vector<std::vector<float>> TorchModel<isCUDA>::forward_multi_ligand_batch(
    const std::vector<float3> &rec_coords, const std::vector<smt> &rec_types,
    const std::vector<std::vector<float3>> &lig_coords_batch,
    const std::vector<std::vector<smt>> &lig_types_batch,
    const std::vector<vec> &centers_batch,
    bool rotate) {

  size_t total_batch_size = lig_coords_batch.size();
  if (total_batch_size == 0) {
    return {};
  }

  torch::NoGradGuard no_grad;  // Disable gradients for inference

  // Force cuBLAS instead of cuBLASLt to avoid CUBLAS_STATUS_NOT_INITIALIZED errors
  if (isCUDA) {
    at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublas);
  }

  // Create receptor CoordinateSet once (reused for all poses)
  CoordinateSet rec = make_coordset(rec_coords, rec_types, rec_typer);

  // Grid dimensions
  long ntypes = rec_typer->num_types() + lig_typer->num_types();
  long gd = gmaker.get_first_dim();

  // Chunk size for memory management (each grid ~12MB for typical settings)
  // Note: Larger batches (512+) cause "double free" crashes - needs investigation
  const size_t MAX_CHUNK_SIZE = 256;

  auto options = torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(isCUDA ? torch::kCUDA : torch::kCPU);

  std::vector<std::vector<float>> all_scores(total_batch_size);

  // Timing accumulators
  double total_voxelize_ms = 0.0;
  double total_nn_inference_ms = 0.0;
  double total_extract_ms = 0.0;

  // Process in chunks
  for (size_t chunk_start = 0; chunk_start < total_batch_size; chunk_start += MAX_CHUNK_SIZE) {
    size_t chunk_end = std::min(chunk_start + MAX_CHUNK_SIZE, total_batch_size);
    size_t chunk_size = chunk_end - chunk_start;

    // Allocate grid tensor for this chunk
    torch::Tensor gtensor = torch::zeros({(long)chunk_size, ntypes, gd, gd, gd}, options);

    // Per-pose voxelization (GPU-accelerated via gmaker.forward)
    auto vox_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunk_size; i++) {
      size_t global_idx = chunk_start + i;

      // Create ligand CoordinateSet for this pose
      CoordinateSet lig = make_coordset(lig_coords_batch[global_idx], lig_types_batch[global_idx], lig_typer);

      // Calculate center (use ligand center if not specified)
      const vec& center = centers_batch[global_idx];
      float3 gcenter;
      if (!isfinite(center.x())) {
        gcenter = lig.center();
      } else {
        gcenter = {center.x(), center.y(), center.z()};
      }

      // Combine receptor and ligand
      CoordinateSet combined(rec, lig);

      // Apply rotation if requested
      libmolgrid::Transform transform(gcenter, 0, rotate);
      if (rotate) {
        transform.forward(combined, combined);
      }

      // Voxelize to grid slice
      Grid<float, 4, isCUDA> out_slice(gtensor[i].data_ptr<float>(), ntypes, gd, gd, gd);
      gmaker.forward(gcenter, combined, out_slice);
    }
    if (isCUDA) cudaDeviceSynchronize();
    auto vox_end = std::chrono::high_resolution_clock::now();
    total_voxelize_ms += std::chrono::duration<double, std::milli>(vox_end - vox_start).count();

    // Run batched model inference for entire chunk
    auto nn_start = std::chrono::high_resolution_clock::now();
    vector<torch::jit::IValue> inputs{gtensor};
    auto result = module.forward(inputs).toTuple()->elements();

    // Extract results
    auto pose_logits = result[0].toTensor();  // (chunk_size, 2)
    auto affinities = result[1].toTensor();   // (chunk_size,)

    torch::Tensor pose_scores;
    if (skip_softmax) {
      pose_scores = pose_logits.index({"...", 1});
    } else {
      pose_scores = torch::softmax(pose_logits, 1).index({"...", 1});
    }

    auto pose_scores_cpu = pose_scores.cpu();
    auto affinities_cpu = affinities.cpu();
    auto nn_end = std::chrono::high_resolution_clock::now();
    total_nn_inference_ms += std::chrono::duration<double, std::milli>(nn_end - nn_start).count();

    auto extract_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunk_size; i++) {
      float pose = pose_scores_cpu[i].item<float>();
      float aff = affinities_cpu[i].item<float>();
      all_scores[chunk_start + i] = {pose, aff, 0.0f};
    }
    auto extract_end = std::chrono::high_resolution_clock::now();
    total_extract_ms += std::chrono::duration<double, std::milli>(extract_end - extract_start).count();

    // Clear GPU cache between chunks
    if (isCUDA) {
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
  }

  // Report timing breakdown
  double total_ms = total_voxelize_ms + total_nn_inference_ms + total_extract_ms;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  CNN batch timing: voxelize=" << total_voxelize_ms/1000.0 << "s ("
            << (100.0*total_voxelize_ms/total_ms) << "%), "
            << "NN=" << total_nn_inference_ms/1000.0 << "s ("
            << (100.0*total_nn_inference_ms/total_ms) << "%), "
            << "extract=" << total_extract_ms/1000.0 << "s ("
            << (100.0*total_extract_ms/total_ms) << "%)\n";

  return all_scores;
}

// Voxelize poses into grid tensors (for ensemble caching)
template <bool isCUDA>
std::vector<torch::Tensor> TorchModel<isCUDA>::voxelize_multi_ligand_batch(
    const std::vector<float3> &rec_coords, const std::vector<smt> &rec_types,
    const std::vector<std::vector<float3>> &lig_coords_batch,
    const std::vector<std::vector<smt>> &lig_types_batch,
    const std::vector<vec> &centers_batch,
    bool rotate,
    double& voxelize_time_ms) {

  size_t total_batch_size = lig_coords_batch.size();
  if (total_batch_size == 0) {
    voxelize_time_ms = 0.0;
    return {};
  }

  torch::NoGradGuard no_grad;

  if (isCUDA) {
    at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublas);
  }

  // Create receptor CoordinateSet once (reused for all poses)
  CoordinateSet rec = make_coordset(rec_coords, rec_types, rec_typer);

  // Grid dimensions
  long ntypes = rec_typer->num_types() + lig_typer->num_types();
  long gd = gmaker.get_first_dim();

  const size_t MAX_CHUNK_SIZE = 256;

  auto options = torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(isCUDA ? torch::kCUDA : torch::kCPU);

  std::vector<torch::Tensor> grid_chunks;
  voxelize_time_ms = 0.0;

  // Process in chunks
  for (size_t chunk_start = 0; chunk_start < total_batch_size; chunk_start += MAX_CHUNK_SIZE) {
    size_t chunk_end = std::min(chunk_start + MAX_CHUNK_SIZE, total_batch_size);
    size_t chunk_size = chunk_end - chunk_start;

    // Allocate grid tensor for this chunk
    torch::Tensor gtensor = torch::zeros({(long)chunk_size, ntypes, gd, gd, gd}, options);

    auto vox_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunk_size; i++) {
      size_t global_idx = chunk_start + i;

      // Create ligand CoordinateSet for this pose
      CoordinateSet lig = make_coordset(lig_coords_batch[global_idx], lig_types_batch[global_idx], lig_typer);

      // Calculate center
      const vec& center = centers_batch[global_idx];
      float3 gcenter;
      if (!isfinite(center.x())) {
        gcenter = lig.center();
      } else {
        gcenter = {center.x(), center.y(), center.z()};
      }

      // Combine receptor and ligand
      CoordinateSet combined(rec, lig);

      // Apply rotation if requested
      libmolgrid::Transform transform(gcenter, 0, rotate);
      if (rotate) {
        transform.forward(combined, combined);
      }

      // Voxelize to grid slice
      Grid<float, 4, isCUDA> out_slice(gtensor[i].data_ptr<float>(), ntypes, gd, gd, gd);
      gmaker.forward(gcenter, combined, out_slice);
    }
    if (isCUDA) cudaDeviceSynchronize();
    auto vox_end = std::chrono::high_resolution_clock::now();
    voxelize_time_ms += std::chrono::duration<double, std::milli>(vox_end - vox_start).count();

    grid_chunks.push_back(gtensor);
  }

  return grid_chunks;
}

// Run NN inference on pre-voxelized grids
template <bool isCUDA>
std::vector<std::vector<float>> TorchModel<isCUDA>::forward_from_grids(
    const std::vector<torch::Tensor>& grid_chunks,
    double& nn_time_ms,
    double& extract_time_ms) {

  if (grid_chunks.empty()) {
    nn_time_ms = 0.0;
    extract_time_ms = 0.0;
    return {};
  }

  torch::NoGradGuard no_grad;

  if (isCUDA) {
    at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublas);
  }

  // Count total poses
  size_t total_batch_size = 0;
  for (const auto& chunk : grid_chunks) {
    total_batch_size += chunk.size(0);
  }

  std::vector<std::vector<float>> all_scores(total_batch_size);
  nn_time_ms = 0.0;
  extract_time_ms = 0.0;

  size_t global_offset = 0;
  for (const auto& gtensor : grid_chunks) {
    size_t chunk_size = gtensor.size(0);

    // Run batched model inference
    auto nn_start = std::chrono::high_resolution_clock::now();
    vector<torch::jit::IValue> inputs{gtensor};
    auto result = module.forward(inputs).toTuple()->elements();

    auto pose_logits = result[0].toTensor();
    auto affinities = result[1].toTensor();

    torch::Tensor pose_scores;
    if (skip_softmax) {
      pose_scores = pose_logits.index({"...", 1});
    } else {
      pose_scores = torch::softmax(pose_logits, 1).index({"...", 1});
    }

    auto pose_scores_cpu = pose_scores.cpu();
    auto affinities_cpu = affinities.cpu();
    auto nn_end = std::chrono::high_resolution_clock::now();
    nn_time_ms += std::chrono::duration<double, std::milli>(nn_end - nn_start).count();

    auto extract_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunk_size; i++) {
      float pose = pose_scores_cpu[i].item<float>();
      float aff = affinities_cpu[i].item<float>();
      all_scores[global_offset + i] = {pose, aff, 0.0f};
    }
    auto extract_end = std::chrono::high_resolution_clock::now();
    extract_time_ms += std::chrono::duration<double, std::milli>(extract_end - extract_start).count();

    global_offset += chunk_size;
  }

  return all_scores;
}

// explicit instantiations
template class TorchModel<true>;
template class TorchModel<false>;
