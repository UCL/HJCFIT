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

#ifndef DCPROGS_LIKELIHOOD_H
#define DCPROGS_LIKELIHOOD_H

#include <DCProgsConfig.h>

#include <vector>
#include <unsupported/Eigen/MatrixFunctions>

#include "qmatrix.h"
#include "errors.h"

namespace DCProgs {

  //! Computes likelihood of a time series.
  //! \param[in] _begin: First interval in the time series. This must be an "open" interval.
  //! \param[in] _end: One past last interval.
  //! \param[in] _g: The likelihood functor. It should have an `af(t_real)` and an `fa(t_real)`
  //!                member function, where the argument is the length of an open or shut interval.
  //! \param[in] _initial: initial occupancies.
  //! \param[in] _final: final occupancies.
  template<class T_INTERVAL_ITERATOR, class T_G>
    t_real chained_likelihood( T_G const & _g, T_INTERVAL_ITERATOR _begin, T_INTERVAL_ITERATOR _end, 
                               t_initvec const &_initial, t_rvector const &_final ) {
      if( (_end - _begin) % 2 != 1 )
        throw errors::Domain("Expected a burst with odd number of intervals");
      t_initvec current = _initial * _g.af(static_cast<t_real>(*_begin));
      while(++_begin != _end) {
        current = current * _g.fa(static_cast<t_real>(*_begin));
        current = current * _g.af(static_cast<t_real>(*(++_begin)));
      }
      return current * _final;
    }

  //! \brief Computes log10-likelihood of a time series.
  //! \details Adds a bit of trickery to take care of exponent. May make this a bit more stable.
  //! \param[in] _begin: First interval in the time series. This must be an "open" interval.
  //! \param[in] _end: One past last interval.
  //! \param[in] _g: The likelihood functor. It should have an `af(t_real)` and an `fa(t_real)`
  //!                member function, where the argument is the length of an open or shut interval.
  //! \param[in] _initial: initial occupancies.
  //! \param[in] _final: final occupancies.
  template<class T_INTERVAL_ITERATOR, class T_G>
    t_real chained_log10_likelihood( T_G const & _g, T_INTERVAL_ITERATOR _begin,
                                     T_INTERVAL_ITERATOR _end, 
                                     t_initvec const &_initial, t_rvector const &_final ) {
      if( (_end - _begin) % 2 != 1 )
        throw errors::Domain("Expected a burst with odd number of intervals");
      t_initvec current = _initial * _g.af(static_cast<t_real>(*_begin));
      t_int exponent(0);
      while(++_begin != _end) {
        current = current * _g.fa(static_cast<t_real>(*_begin));
        current = current * _g.af(static_cast<t_real>(*(++_begin)));
        t_real const max_coeff = current.array().abs().maxCoeff();
        if(max_coeff > 1e50) {
          current  *= 1e-50;
          exponent += 50;
        } else if(max_coeff < -50) {
          current  *= 1e+50;
          exponent -= 50;
        }
      }
      return std::log10(current * _final) + exponent;
    }


  //! \brief A functor with which to optimize a QMatrix
  //! \details This functor takes as input a qmatrix and returns the likelihood for a given set of
  //!          bursts. It is, in practice, a convenience object with which to perform likelihood
  //!          optimization.
  class MSWINDOBE Log10Likelihood {
    public:
      //! Set of bursts for which to compute the likelihood.
      t_Bursts bursts;
      //! Number of open states.
      t_uint nopen;
      //! Max length of missed events
      t_real tau;
      //! \brief tcrit. 
      //! \detail If negative or null, will use equilibrium occupancies rather than CHS occupancies.
      t_real tcritical;
      //! Number of intervals for which to compute exact result.
      t_uint nmax;
      //! Tolerance for root finding.
      t_real xtol;
      //! Tolerance for root finding.
      t_real rtol;
      //! Maximum number of iterations for root finding.
      t_uint itermax;


      //! Constructor
      Log10Likelihood   ( t_Bursts const &_bursts, t_uint _nopen, t_real _tau,
                          t_real _tcritical=-1e0, t_uint _nmax=2, t_real _xtol=1e-10,
                          t_real _rtol=1e-10, t_uint _itermax=100 ) 
                      : bursts(_bursts), nopen(_nopen), tau(_tau), tcritical(_tcritical),
                        nmax(_nmax), xtol(_xtol), rtol(_rtol), itermax(_itermax) {}
     
      //! Computes likelihood for each burst in separate value.
      t_rvector vector(t_rmatrix const &_Q) const { return vector(QMatrix(_Q, nopen)); }
      //! Computes likelihood for each burst in separate value.
      t_rvector vector(QMatrix const &_Q) const;
      //! Log-likelihood 
      t_real operator()(t_rmatrix const &_Q) const { return operator()(QMatrix(_Q, nopen)); }
      //! Log-likelihood 
      t_real operator()(QMatrix const &_Q) const;
  };
  //! Dumps likelihood to stream.
  MSWINDOBE std::ostream& operator<<(std::ostream& _stream, Log10Likelihood const & _self);
  //! Dumps burst to stream.
  MSWINDOBE std::ostream& operator<<(std::ostream& _stream, t_Burst const & _self);
  //! Dumps bursts to stream.
  MSWINDOBE std::ostream& operator<<(std::ostream& _stream, t_Bursts const & _self);
}

#endif 

