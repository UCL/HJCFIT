/***********************
    DCProgs computes missed-events likelihood as described in
    Hawkes, Jalali and Colquhoun (1990, 1992)

    Copyright (C) 2013  University College London

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
************************/

#include <DCProgsConfig.h>

#include <sstream>
#include <iostream>
#ifdef OPENMP_FOUND
# include <omp.h>
#endif

#include "likelihood.h"
#include "occupancies.h"
#include "missed_eventsG.h"

namespace DCProgs {

  MSWINDOBE std::ostream& operator<<(std::ostream& _stream, t_Bursts const & _self) {
    _stream << "Bursts:\n"
            << "-------\n"
            << "  [ ";
    for(t_Bursts::size_type i(0); i + 1 < _self.size(); ++i) _stream << _self[i] << ",\n    ";
    _stream << _self.back() << " ]\n";
    return _stream;
  }

  MSWINDOBE std::ostream& operator<<(std::ostream& _stream, t_Burst const & _self) {
    Eigen::Map<const t_initvec> vector(&(_self[0]), _self.size());
    return _stream << vector.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ",", ",\n",
                                                    "", "", "[", "]" )); 
  }

  MSWINDOBE std::ostream& operator<<(std::ostream& _stream, Log10Likelihood const & _self) {
    
    _stream << "Log10 Likelihood:\n"
            << "=================\n\n" 
            << "  * Number of open states: " << _self.nopen << "\n"
            << "  * Resolution time tau: " << _self.tau << "\n";
    if(DCPROGS_ISNAN(_self.tcritical)  or _self.tcritical <= 0e0)
         _stream << "  * Using equilibrium occupancies.\n";
    else _stream << "  * Using CHS occupancies with tcrit: "  << _self.tcritical << "\n";
    _stream << "  * Exact events computed for: t < "
            << _self.nmax << " tau\n\n"
            << _self.bursts
            << "\nRoot Finding:\n"
            << "-------------\n"
            << "  * Tolerance criteria: " << _self.xtol << ", " << _self.rtol << "\n"
            << "  * Maximum number of iterations: " << _self.itermax << "\n";
    return _stream;
  }

  namespace {
    t_real one_burst(
            MissedEventsG const &_eG, t_Burst const &_burst,
            size_t _nshut, t_initvec const &_initial, t_rvector const &_final) {
#     ifdef OPENMP___FOUND
        if(_burst.size() % 2 != 1)
          throw errors::Domain("Expected a burst with odd number of intervals");

        t_initvec allvec = t_initvec::Zero(_nshut);
        t_real coeff = 0;
#       pragma omp parallel shared(allvec) reduction(+:coeff)
        {
          size_t const nthreads = omp_get_num_threads();
          size_t const id = omp_get_thread_num();
          size_t const N = (_burst.size() - 1) >> 1;
          size_t const leftover = N % nthreads;
          size_t const addsome = id < leftover ? (std::min(id, leftover) << 1): 0;
          size_t const per_thread = (N / nthreads) << 1;
          size_t const first = 1 + per_thread * id + addsome;
          size_t const end = first + per_thread * (id+1) + size_t(id < leftover ? 2: 0);
          auto partial =
              chained_log10_likelihood(_eG, _burst.begin() + first, _burst.begin() + end);
          if(id == 0) allvec = _initial * _eG.af(static_cast<t_real>(_burst.front()));

          coeff = partial.second;
#         pragma omp for ordered
          for(size_t i = 0; i < nthreads; ++i) {
#           pragma omp order
            allvec = allvec * partial.first;
          }
        }
        return std::log10(allvec * _final) + coeff;
#     else
        return chained_log10_likelihood(_eG, _burst.begin(), _burst.end(), _initial, _final);
#     endif
    }
  }

  t_real Log10Likelihood::operator()(QMatrix const &_matrix) const {
    MissedEventsG const eG = MissedEventsG( _matrix, tau, nmax, xtol, rtol, itermax,
                                            lower_bound, upper_bound );
    bool const eq_vector = DCPROGS_ISNAN(tcritical) or tcritical <= 0;

    t_rvector final;

    if(eq_vector)
        final = t_rmatrix::Ones(_matrix.nshut(),1);
    else
        final = CHS_occupancies(eG, tcritical, false).transpose();

    t_initvec const initial = eq_vector ? occupancies(eG): CHS_occupancies(eG, tcritical);
    t_real result(0);
#   pragma parallel for shared(bursts, eG, _matrix, initial, final) reduction(+:result)
    for(size_t i=0; i < bursts.size(); ++i)
      if(bursts[i].size() >= 1)
          result += one_burst(eG, bursts[i], _matrix.nshut(), initial, final);
    return result;
  }

  t_rvector Log10Likelihood::vector(QMatrix const &_matrix) const {
    MissedEventsG const eG = MissedEventsG( _matrix, tau, nmax, xtol, rtol, itermax,
                                            lower_bound, upper_bound );
    bool const eq_vector = DCPROGS_ISNAN(tcritical) or tcritical <= 0;

    t_rvector final;

    if(eq_vector)
        final = t_rmatrix::Ones(_matrix.nshut(),1);
    else
        final = CHS_occupancies(eG, tcritical, false).transpose();

    t_initvec const initial = eq_vector ? occupancies(eG): CHS_occupancies(eG, tcritical);
    t_rvector result = t_rvector::Ones(bursts.size());
#   pragma parallel for shared(bursts, eG, _matrix, initial, final, result)
    for(size_t i=0; i < bursts.size(); ++i)
      if(bursts[i].size() >= 1)
          result(i) = one_burst(eG, bursts[i], _matrix.nshut(), initial, final);
    return result;
  }
}
