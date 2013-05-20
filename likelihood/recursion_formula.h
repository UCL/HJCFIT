#ifndef DCPROGS_LIKELIHOOD_RECURSION_FORMULA_H
#define DCPROGS_LIKELIHOOD_RECURSION_FORMULA_H

#include <DCProgsConfig.h>

namespace DCProgs {

  namespace details {
    template<class T, class T_ZERO> 
      typename T::t_element general(T & _C, t_int _i, t_int _m, t_int _l, T_ZERO const &_zero);
    template<class T, class T_ZERO> 
      typename T::t_element lzero(T & _C, t_int _i, t_int _m, T_ZERO const &_zero);
  }

  //! \brief Obtains _C[_i, _m, _l] if prior terms are known.
  //! \details This function implements the recursion with as few requirements as possible. The
  //!          objective is to make the recursion clearer and testing easier. It also means we
  //!          separate concerns, such as caching prior results, or the possibility of using
  //!          arbitrary precision types.
  //!
  //!          Implements eq 3.18 from Hawkes, Jalali, Colquhoun (1990).
  //! \tparam T: A type with the following form:
  //!    \code{.cpp}
  //!      class T {
  //!        //! Type of the element
  //!        typedef t_element;
  //!
  //!        //! Returns (prior) element in recursion
  //!        t_element operator()(t_int _i, t_int _j, t_int _m);
  //!        //! \brief Returns D objects, e.g. \f$A_{iAF}e^{Q_{FF}\tau}Q_{FA}\f$.
  //!        auto getD(t_int _i) const;
  //!        //! Returns specific eigenvalue of \f$Q\f$.
  //!        auto get_eigval(t_int _i) const;
  //!        //! Returns number of eigenvalues.
  //!        t_int nbeigval(t_int _i) const;
  //!      };
  //!    \endcode
  //! \tparam T_ZERO: Type of a functor.
  //! \param _C: The object over which the recursion is perfomed.
  //! \param _i: An integer
  //! \param _j: An integer
  //! \param _l: An integer
  //! \parma _zero: A functor used to initialise intermediate objects.
  template<class T, class T_ZERO> 
    typename T::t_element recursion_formula( T & _C, t_int _i, t_int _m, t_int _l,
                                             T_ZERO const &_zero ) {
      
      // first, deal with _m == 0 and _l == 0 case.
      if(_m == 0 and _l == 0) return _C(_i, 0, 0);
      // then deals with two _l = 0 case
      if(_l == 0) return details::lzero(_C, _i, _m, _zero);
      // then deals with _l == _m
      if(_l == _m) return _C.getD(_i) * _C(_i, _m-1, _m-1) / t_real(_m);

      return details::general(_C, _i, _m, _l, _zero);
    }

  namespace details {

    template<class T, class T_ZERO> 
      typename T::t_element lzero(T & _C, t_int _i, t_int _m, T_ZERO const &_zero) {

        auto result = _zero();
        auto Di = _C.getD(_i);
        auto lambda_i = _C.get_eigvals(_i); 

        for(t_int j(0); j < _C.nbeigvals(); ++j) {
          if(_i == j) continue;

          auto Dj = _C.getD(j);
          auto lambda_j = _C.get_eigvals(j);

          t_real const lambda_invdiff(1e0/(lambda_j - lambda_i)); 
          t_real factor(lambda_invdiff); 

          for(t_int r(0), sign(1); r < _m; ++r, sign = -sign) {
            result += (Di * _C(j, _m-1, r) - sign * Dj * _C(_i, _m-1, r)) * factor;
            factor *= lambda_invdiff * (r+1);
          }
        } // loop over j
  
        return result;
      }

    template<class T, class T_ZERO> 
      typename T::t_element general(T & _C, t_int _i, t_int _m, t_int _l, T_ZERO const &_zero) {

        typename T::t_element result(_C.getD(_i) * _C(_i, _m-1, _l-1) / t_real(_l));
        for(t_int j(0); j < _C.nbeigvals(); ++j) {
          if(_i == j) continue;

          t_real const lambda(1e0/(_C.get_eigvals(_i)-_C.get_eigvals(j))); 
          t_real factor(lambda); 
          auto intermediate = _zero();
          for(t_int r(_l); r < _m; ++r) {
            intermediate += _C(_i, _m-1, r) * factor;
            factor *= lambda * (r+1);
          }
          result -= _C.getD(j) * intermediate;
        } // loop over j
  
        return result;
      }
  }
}
#endif 