/*

 Copyright (c) 2006-2010, The Scripps Research Institute

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 Author: Dr. Oleg Trott <ot14@columbia.edu>,
 The Olson Lab,
 The Scripps Research Institute

 */

#include "parallel.h"
#include "parallel_mc.h"
#include "coords.h"
#include "parallel_progress.h"
#include "non_cache_cnn.h"
#include "user_opts.h"

struct parallel_mc_task {
    model m;
    output_container out;
    rng generator;
    parallel_mc_task(const model& m_, int seed)
        : m(m_), generator(static_cast<rng::result_type>(seed)) {
    }
};

typedef boost::ptr_vector<parallel_mc_task> parallel_mc_task_container;

struct parallel_mc_aux {
    const monte_carlo* mc;
    const precalculate* p;
    igrid* ig;
    const vec* corner1;
    const vec* corner2;
    parallel_progress* pg;
    grid* user_grid;
    non_cache* nc;
    parallel_mc_aux(const monte_carlo* mc_, const precalculate* p_, igrid* ig_,
        const vec* corner1_, const vec* corner2_, parallel_progress* pg_,
        grid* user_grid_, non_cache* nc_)
        : mc(mc_), p(p_), ig(ig_), corner1(corner1_), corner2(corner2_),
            pg(pg_), user_grid(user_grid_), nc(nc_) {
    }

    void operator()(parallel_mc_task& t) const {
      const non_cache_cnn* cnn = dynamic_cast<const non_cache_cnn*>(nc);
      if (cnn && cnn->get_scorer().options().cnn_scoring>CNNrefinement) {
        std::shared_ptr<DLScorer> cnn_scorer = cnn->get_scorer().fresh_copy();
        const precalculate* p = cnn->get_precalculate();
        szv_grid_cache gridcache(t.m, p->cutoff_sqr());
        non_cache_cnn new_cnn(gridcache, cnn->get_grid_dims(), p,
            cnn->getSlope(), *cnn_scorer);
        if (cnn_scorer->options().cnn_scoring==CNNmetropolisrefine||cnn_scorer->options().cnn_scoring==CNNmetropolisrescore)
        {
            (*mc)(t.m, t.out, *p, *ig, *corner1, *corner2, pg, t.generator,
                *user_grid, new_cnn);
        }
        else{//cnn_scoring=CNNall
            (*mc)(t.m, t.out, *p, new_cnn, *corner1, *corner2, pg, t.generator,
                *user_grid,new_cnn);
        }
      } else
        (*mc)(t.m, t.out, *p, *ig, *corner1, *corner2, pg, t.generator,
            *user_grid, *ig);
    }
};

//TODO: null model.gdata pointers at task exit

void merge_output_containers(const output_container& in, output_container& out,
    fl min_rmsd, sz max_size) {
  VINA_FOR_IN(i, in)
    add_to_output_container(out, in[i], min_rmsd, max_size);
}

void merge_output_containers(const parallel_mc_task_container& many,
    output_container& out, fl min_rmsd, sz max_size) {
  min_rmsd = 2; // FIXME? perhaps it's necessary to separate min_rmsd during search and during output?
  VINA_FOR_IN(i, many) {
    merge_output_containers(many[i].out, out, min_rmsd, max_size);
  }
  out.sort();
}

void parallel_mc::operator()(const model& m, output_container& out,
    const precalculate& p, igrid& ig, const vec& corner1, const vec& corner2,
    rng& generator, grid& user_grid, non_cache& nc) const {
  parallel_progress pp;
  parallel_mc_aux parallel_mc_aux_instance(&mc, &p, &ig, &corner1, &corner2,
      (display_progress ? (&pp) : NULL), &user_grid, &nc);
  parallel_mc_task_container task_container;
  VINA_FOR(i, num_tasks)
    task_container.push_back(
        new parallel_mc_task(m, random_int(0, 1000000, generator)));
  if (display_progress) pp.init(num_tasks * mc.num_steps);

  auto thread_init = [&]()
  {
    initializeCUDA(m.gdata.device_id); //harmless to do if there isn't a gpu
  };

  parallel_iter<parallel_mc_aux,
  parallel_mc_task_container, parallel_mc_task,
      decltype(thread_init), true> parallel_iter_instance(
      &parallel_mc_aux_instance, num_threads, thread_init);
  parallel_iter_instance.run(task_container);

  merge_output_containers(task_container, out, mc.min_rmsd, mc.num_saved_mins);

}
