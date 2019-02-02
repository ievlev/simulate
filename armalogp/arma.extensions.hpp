////////////////////////////////////////////////////////////////////////////////////
// The MIT License (MIT)                                                          //
//                                                                                //
// Copyright (c) 2015 Whit Armstrong                                              //
//                                                                                //
// Permission is hereby granted, free of charge, to any person obtaining a copy   //
// of this software and associated documentation files (the "Software"), to deal  //
// in the Software without restriction, including without limitation the rights   //
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      //
// copies of the Software, and to permit persons to whom the Software is          //
// furnished to do so, subject to the following conditions:                       //
//                                                                                //
// The above copyright notice and this permission notice shall be included in all //
// copies or substantial portions of the Software.                                //
//                                                                                //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    //
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  //
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  //
// SOFTWARE.                                                                      //
////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <armadillo>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/factorials.hpp>

namespace arma {

  // lgamma
  class eop_lgamma : public eop_core<eop_lgamma> {};

  template<> template<typename eT> arma_hot arma_pure arma_inline eT
  eop_core<eop_lgamma>::process(const eT val, const eT  ) {
    return boost::math::lgamma(val);
  }

  // Base
  template<typename T1>
  arma_inline
  const eOp<T1, eop_lgamma> lgamma(const Base<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_lgamma>(A.get_ref());
  }

  // BaseCube
  template<typename T1>
  arma_inline
  const eOpCube<T1, eop_lgamma> lgamma(const BaseCube<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOpCube<T1, eop_lgamma>(A.get_ref());
  }




  // factln
  double factln(const int i) {
    static std::vector<double> factln_table;

    if(i < 0) {
      return -std::numeric_limits<double>::infinity();
    }

    if(i > 100) {
      return boost::math::lgamma(static_cast<double>(i) + 1);
    }

    if(factln_table.size() < static_cast<size_t>(i+1)) {
      for(int j = factln_table.size(); j < (i+1); j++) {
        factln_table.push_back(std::log(boost::math::factorial<double>(static_cast<double>(j))));
      }
    }
    //for(auto v : factln_table) { std::cout << v << "|"; }  std::cout << std::endl;
    return factln_table[i];
  }

  class eop_factln : public eop_core<eop_factln> {};

  template<> template<typename eT> arma_hot arma_pure arma_inline eT
  eop_core<eop_factln>::process(const eT val, const eT  ) {
    return factln(val);
  }

  // Base
  template<typename T1>
  arma_inline
  const eOp<T1, eop_factln> factln(const Base<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_factln>(A.get_ref());
  }

  // BaseCube
  template<typename T1>
  arma_inline
  const eOpCube<T1, eop_factln> factln(const BaseCube<typename T1::elem_type,T1>& A) {
    arma_extra_debug_sigprint();
    return eOpCube<T1, eop_factln>(A.get_ref());
  }

  // cube
  //! element-wise multiplication of BaseCube objects with same element type
  template<typename T1, typename T2>
  arma_inline
  const eGlueCube<T1, T2, eglue_schur>
  schur_prod
  (
   const BaseCube<typename T1::elem_type,T1>& X,
   const BaseCube<typename T1::elem_type,T2>& Y
   )
  {
    arma_extra_debug_sigprint();
    return eGlueCube<T1, T2, eglue_schur>(X.get_ref(), Y.get_ref());
  }

  //! element-wise multiplication of BaseCube objects with different element types
  template<typename T1, typename T2>
  inline
  const mtGlueCube<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_schur>
  schur_prod
  (
   const BaseCube< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T1_result, T1>& X,
   const BaseCube< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T2_result, T2>& Y
   )
  {
    arma_extra_debug_sigprint();
    typedef typename T1::elem_type eT1;
    typedef typename T2::elem_type eT2;
    typedef typename promote_type<eT1,eT2>::result out_eT;
    promote_type<eT1,eT2>::check();
    return mtGlueCube<out_eT, T1, T2, glue_mixed_schur>( X.get_ref(), Y.get_ref() );
  }

  // matrix
  template<typename T1, typename T2>
  arma_inline
  const eGlue<T1, T2, eglue_schur>
  schur_prod(const Base<typename T1::elem_type,T1>& X, const Base<typename T1::elem_type,T2>& Y) {
    arma_extra_debug_sigprint();
    return eGlue<T1, T2, eglue_schur>(X.get_ref(), Y.get_ref());
  }

  //! element-wise multiplication of Base objects with different element types
  template<typename T1, typename T2>
  inline
  const mtGlue<typename promote_type<typename T1::elem_type, typename T2::elem_type>::result, T1, T2, glue_mixed_schur>
  schur_prod
  (
   const Base< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T1_result, T1>& X,
   const Base< typename force_different_type<typename T1::elem_type, typename T2::elem_type>::T2_result, T2>& Y
   )
  {
    arma_extra_debug_sigprint();
    typedef typename T1::elem_type eT1;
    typedef typename T2::elem_type eT2;
    typedef typename promote_type<eT1,eT2>::result out_eT;
    promote_type<eT1,eT2>::check();
    return mtGlue<out_eT, T1, T2, glue_mixed_schur>( X.get_ref(), Y.get_ref() );
  }


  //! Base * scalar
  template<typename T1>
  arma_inline
  const eOp<T1, eop_scalar_times>
  schur_prod
  (const Base<typename T1::elem_type,T1>& X, const typename T1::elem_type k)
  {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_scalar_times>(X.get_ref(),k);
  }

  //! scalar * Base
  template<typename T1>
  arma_inline
  const eOp<T1, eop_scalar_times>
  schur_prod
  (const typename T1::elem_type k, const Base<typename T1::elem_type,T1>& X)
  {
    arma_extra_debug_sigprint();
    return eOp<T1, eop_scalar_times>(X.get_ref(),k);  // NOTE: order is swapped
  }

  double schur_prod(const int x, const double y) { return x * y; }
  double schur_prod(const double x, const int y) { return x * y; }
  double schur_prod(const double& x, const double& y) { return x * y; }
  double schur_prod(const int& x, const int& y) { return x * y; }

  // insert an 'any' function for bools into the arma namespace
  bool any(const bool x) {
    return x;
  }

  bool vectorise(bool x) {
    return x;
  }
} // namespace arma
