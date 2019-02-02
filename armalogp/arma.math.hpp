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

#include <stdexcept>
#include <armadillo>

namespace armalogp {

  static inline double square(double x) {
    return x*x;
  }

  static inline int square(int x) {
    return x*x;
  }

  double cholesky_determinant(const arma::mat& R) {
    return arma::prod(square(R.diag()));
  }

  double mahalanobis(const arma::vec& x, const arma::vec& mu, const arma::mat& sigma) {
    const arma::vec err = x - mu;
    return arma::as_scalar(err.t() * sigma.i() * err);
  }

  double mahalanobis(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& sigma) {
    const arma::rowvec err = x - mu;
    return arma::as_scalar(err * sigma.i() * err.t());
  }

  double mahalanobis_chol(const arma::rowvec& x, const arma::rowvec& mu, const arma::mat& R) {
    const arma::rowvec err = x - mu;
    const arma::mat Rinv(inv(trimatl(R)));
    return arma::as_scalar(err * Rinv * Rinv.t() * err.t());
  }

} // namespace armalogp

