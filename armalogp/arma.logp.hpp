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
#include <armalogp/arma.extensions.hpp>
#include <armalogp/arma.math.hpp>

namespace armalogp {

  template<double LOGF(double), typename T, typename U, typename V>
  double normal_logp(const T& x, const U& mu, const V& tau) {
    return arma::accu(0.5*LOGF(0.5*tau/arma::math::pi()) - 0.5 * arma::schur_prod(tau, square(x - mu)));
  }

  template<double LOGF(double), typename T, typename U, typename V>
  double uniform_logp(const T& x, const U& lower, const V& upper) {
    return (arma::any(arma::vectorise(x < lower)) || arma::any(arma::vectorise(x > upper))) ? -std::numeric_limits<double>::infinity() : -arma::accu(LOGF(upper - lower));
  }

  template<double LOGF(double), typename T, typename U, typename V>
  double gamma_logp(const T& x, const U& alpha, const V& beta) {
    return arma::any(arma::vectorise(x < 0)) ?
      -std::numeric_limits<double>::infinity() :
      arma::accu(arma::schur_prod((alpha - 1.0),LOGF(x)) - arma::schur_prod(beta,x) - lgamma(alpha) + arma::schur_prod(alpha,LOGF(beta)));
  }

  template<double LOGF(double), typename T, typename U, typename V>
  double beta_logp(const T& x, const U& alpha, const V& beta) {
    const double one = 1.0;
    return arma::any(arma::vectorise(x <= 0)) || arma::any(arma::vectorise(x >= 1)) || arma::any(arma::vectorise(alpha <= 0)) || arma::any(arma::vectorise(beta <= 0)) ?
      -std::numeric_limits<double>::infinity() :
      arma::accu(lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta) + arma::schur_prod((alpha-one),LOGF(x)) + arma::schur_prod((beta-one),LOGF(one-x)));
  }

  template<double LOGF(double)>
  double categorical_logp(const arma::ivec& x, const arma::mat& p) {
    if(arma::any(arma::vectorise(p <= 0)) || arma::any(arma::vectorise(p >= 1)) || arma::any(arma::vectorise(x < 0)) || arma::any(arma::vectorise(x >= p.n_cols))) {
      return -std::numeric_limits<double>::infinity();
    }
    // replace w/ call to p.elems later
    double ans(0);
    for(unsigned int i = 0; i < x.n_rows; i++) {
      ans += LOGF(p(i,x[i]));
    }
    return ans;
  }

  template<double LOGF(double)>
  double categorical_logp(const arma::ivec& x, const arma::vec& p) {
    if(arma::any(arma::vectorise(p <= 0)) || arma::any(arma::vectorise(p >= 1)) || arma::any(arma::vectorise(x < 0)) || arma::any(arma::vectorise(x >= p.n_elem))) {
      return -std::numeric_limits<double>::infinity();
    }
    // replace w/ call to p.elems later
    double ans(0);
    for(unsigned int i = 0; i < x.n_rows; i++) {
      ans += LOGF(p(x[i]));
    }
    return ans;
  }

  template<double LOGF(double)>
  double categorical_logp(const int x, const arma::vec& p) {
    return LOGF(p[x]);
  }

  template<double LOGF(double), typename T, typename U, typename V>
  double binomial_logp(const T& x, const U& n, const V& p) {
    if(arma::any(arma::vectorise(p <= 0)) || arma::any(arma::vectorise(p >= 1)) || arma::any(arma::vectorise(x < 0))  || arma::any(arma::vectorise(x > n))) {
      return -std::numeric_limits<double>::infinity();
    }
    return arma::accu(arma::schur_prod(x,LOGF(p)) + arma::schur_prod((n-x),LOGF(1-p)) + arma::factln(n) - arma::factln(x) - arma::factln(n-x));
  }

  template<double LOGF(double), typename T, typename U>
  double bernoulli_logp(const T& x, const U& p) {
    if( arma::any(arma::vectorise(p <= 0)) || arma::any(arma::vectorise(p >= 1)) || arma::any(arma::vectorise(x < 0))  || arma::any(arma::vectorise(x > 1)) ) {
      return -std::numeric_limits<double>::infinity();
    } else {
      return arma::accu(arma::schur_prod(x,LOGF(p)) + arma::schur_prod((1-x), LOGF(1-p)));
    }
  }

  template<double LOGF(double), typename T, typename U>
  double poisson_logp(const T& x, const U& mu) {
    if( arma::any(arma::vectorise(mu < 0)) || arma::any(arma::vectorise(x < 0))) {
      return -std::numeric_limits<double>::infinity();
    } else {
      return arma::accu(schur(x,LOGF(mu)) - mu - factln(x));
    }
  }

  template<double LOGF(double), typename T, typename U>
  double exponential_logp(const T& x, const U& lambda) {
    return arma::accu(LOGF(lambda) - arma::schur_prod(lambda, x));
  }

  template<double LOGF(double), typename T, typename U>
  double multivariate_normal_chol_logp(const T& x, const U& mu, const arma::mat& R) {
    static double log_2pi = LOGF(2 * arma::math::pi());
    double ldet = LOGF(cholesky_determinant(R));
    return -0.5 * (x.n_elem * log_2pi + ldet + mahalanobis_chol(x,mu,R));
  }

  // sigma denotes cov matrix rather than precision matrix
  template<double LOGF(double), typename T, typename U>
  double multivariate_normal_sigma_logp(const T& x, const U& mu, const arma::mat& sigma) {
    arma::mat R;
    bool chol_succeeded = chol(R,sigma);
    if(!chol_succeeded) { return -std::numeric_limits<double>::infinity(); }

    return multivariate_normal_chol_logp<LOGF>(x, mu, R);
  }

  // sigma denotes cov matrix rather than precision matrix
  template<double LOGF(double)>
  double multivariate_normal_sigma_logp(const arma::mat& x, const arma::vec& mu, const arma::mat& sigma) {
    arma::mat R;
    bool chol_succeeded = chol(R,sigma);
    if(!chol_succeeded) { return -std::numeric_limits<double>::infinity(); }
    const arma::rowvec mu_r = mu.t();
    double ans(0);
    for(size_t i = 0; i < x.n_rows; i++) {
      ans += multivariate_normal_chol_logp<LOGF>(x.row(i), mu_r, R);
    }
    return ans;
  }

  template<double LOGF(double)>
  double multivariate_normal_chol_logp(const arma::mat& x, const arma::vec& mu, const arma::mat& R) {
    const arma::rowvec mu_r = mu.t();
    double ans(0);
    for(size_t i = 0; i < x.n_rows; i++) {
      ans += multivariate_normal_chol_logp<LOGF>(x.row(i), mu_r, R);
    }
    return ans;
  }

  template<double LOGF(double)>
  double wishart_logp(const arma::mat& X, const arma::mat& tau, const unsigned int n) {
    if(X.n_cols != X.n_rows || tau.n_cols != tau.n_rows || X.n_cols != tau.n_rows || X.n_cols > n) { return -std::numeric_limits<double>::infinity(); }
    const double lg2 = LOGF(2.0);
    const int k = X.n_cols;
    const double dx(arma::det(X));
    const double db(arma::det(tau));
    if(dx <= 0 || db <= 0) { return -std::numeric_limits<double>::infinity(); }

    const double ldx(LOGF(dx));
    const double ldb(LOGF(db));
    const arma::mat bx(X * tau);
    const double tbx = arma::trace(bx);

    double cum_lgamma(0);
    for(size_t i = 0; i < X.n_rows; ++i) {
      cum_lgamma += lgamma((n + 1)/2.0);
    }
    return (n - k - 1)/2 * ldx + (n/2.0)*ldb - 0.5*tbx - (n*k/2.0)*lg2 - cum_lgamma;
  }

} // namespace armalogp

