#pragma once
#include "common.h"
#include "random.h"
#include <string>

//enum options and their parsers
enum ApproxType
{
  LinearApprox, SplineApprox, Exact, GPU
};

std::istream& operator>>(std::istream& in, ApproxType& type);

enum pose_sort_order {
  CNNscore,
  CNNaffinity,
  Energy
};

//for reading in as a commandline option
std::istream& operator>>(std::istream &in, pose_sort_order &sort_order);


enum cnn_scoring_level {
  CNNnone, //don't use CNN
  CNNrescore, // use CNN only for final scoring and ranking
  CNNrefinement, // use CNN only for minimization
  CNNmetropolisrescore, // use CNN for MC and final scoring
  CNNmetropolisrefine, // use CNN for MC and minimization
  CNNall // use CNN everywhere
};

std::istream& operator>>(std::istream &in, cnn_scoring_level &cnn_level);


struct cnn_options {
    //stores options associated with cnn scoring
    std::vector<std::string> cnn_models; //path(s) to model file
    std::string cnn_recmap; //optional file specifying receptor atom typing to channel map
    std::string cnn_ligmap; //optional file specifying ligand atom typing to channel map
    std::vector<std::string> cnn_model_names; // name of builtin model
    vec cnn_center;
    unsigned cnn_rotations; //do we want to score multiple orientations?
    cnn_scoring_level cnn_scoring;
    double subgrid_dim;
    fl empirical_weight; //weight for scaling and merging potentials
    bool gradient_check;
    bool mix_emp_force;//merge empirical and CNN minus forces
    bool mix_emp_energy;//merge empirical and CNN energy
    bool verbose;

    unsigned seed; //random seed

    cnn_options()
        : cnn_center(NAN, NAN, NAN),
            cnn_rotations(0), cnn_scoring(CNNrescore),
            subgrid_dim(0.0), gradient_check(false), 
            verbose(false), mix_emp_force(false),mix_emp_energy(false),empirical_weight(1.0),seed(0) {
    }
};

//just a collection of user-specified configurations
struct user_settings {
    sz num_modes;
    fl out_min_rmsd;
    fl forcecap;
    int seed;
    int verbosity;
    int cpu;
    int device; //gpu number

    int exhaustiveness;
    int num_mc_steps;
    int max_mc_steps;
    int num_mc_saved;
    fl temperature;
    pose_sort_order sort_order;

    bool score_only;
    bool randomize_only;
    bool local_only;
    bool dominimize;
    bool include_atom_info;
    bool cnn_cpu; // force CNN scoring to run on CPU instead of GPU
    bool no_lig;
    bool gpu; //use GPU for docking (parallel BFGS instead of MC)
    bool cpu_grid; //use grid-based scoring for CPU local_only (to match GPU behavior)
    int bfgs_iterations; //max BFGS iterations when using --gpu
    bool verbose_grad; //output gradient values for debugging
    bool direct_pairwise; //use direct pairwise scoring with LUT instead of grid interpolation (GPU only)
    int batch_size; //target total poses per GPU batch for multi-ligand batch docking
    bool no_batch; //disable batch docking (process one ligand at a time)
    bool fast_embed; //use fast template-based 3D coordinate generation for SMILES
    bool no_rdkit_smiles; //use OpenBabel instead of RDKit for SMILES 3D generation
    int parallel_embed; //number of conformers to generate in parallel for SMILES
    bool skip_torsion_randomize; //skip torsion randomization (for debugging)
    fl prune_rms_thresh; //RMSD threshold for pruning similar conformers during embedding
    sz cnn_refine_mult; //multiplier for num_modes to determine poses sent to CNN scoring

    cnn_options cnnopts;

    //reasonable defaults
    user_settings()
        :  num_modes(9), out_min_rmsd(1), forcecap(1000),
            seed(auto_seed()), verbosity(1), cpu(1), device(0),
            exhaustiveness(10), num_mc_steps(0), max_mc_steps(0), num_mc_saved(50), temperature(0),
            sort_order(CNNscore), score_only(false),
            randomize_only(false), local_only(false), dominimize(false),
            include_atom_info(false), cnn_cpu(false), no_lig(false),
            gpu(false), cpu_grid(false), bfgs_iterations(50), verbose_grad(false),
            direct_pairwise(false), batch_size(50000), no_batch(false), fast_embed(false),
            no_rdkit_smiles(false), parallel_embed(0), skip_torsion_randomize(false),
            prune_rms_thresh(0), cnn_refine_mult(20) {

    }
};

