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

#include "quasi_newton.h"
#include "bfgs.h"

struct quasi_newton_aux {
    model* m;
    const precalculate* p;
    igrid* ig;
    const vec v;
    const grid* user_grid;
    quasi_newton_aux(model* m_, const precalculate* p_, igrid* ig_,
        const vec& v_, const grid* user_grid_)
        : m(m_), p(p_), ig(ig_), v(v_), user_grid(user_grid_) {
    }

    vec get_center() const {
      return ig->get_center();
    }

    fl operator()(const conf& c, change& g) {
      return m->eval_deriv(*p, *ig, v, c, g, *user_grid);
    }
};

void quasi_newton::operator()(model& m, const precalculate& p, igrid& ig,
    output_type& out, change& g, const vec& v, const grid& user_grid) const {
  // g must have correct size
  quasi_newton_aux aux(&m, &p, &ig, v, &user_grid);
  fl res = 0;
  if (params.type == minimization_params::Simple)
    res = simple_gradient_ascent(aux, out.c, g, average_required_improvement,
        params);
  else
    res = bfgs(aux, out.c, g, average_required_improvement, params);
  out.e = res;
}

