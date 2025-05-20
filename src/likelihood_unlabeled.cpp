// src/likelihood_unlabeled.cpp
#include <cmath>             // for log(), exp(), fabs(), M_PI           // for dgamma(), dbeta()
#include <R_ext/Error.h>     // for Rf_error()
#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() () {
  // ---- Data from R
  DATA_VECTOR(Y);
  DATA_MATRIX(Xhat);
  DATA_INTEGER(homoskedastic);
  DATA_INTEGER(dist_code);

  // Optional extra distribution parameters
  DATA_SCALAR(nu);      // for Student‐t
  DATA_SCALAR(gshape);  // for Gamma(shape, scale)
  DATA_SCALAR(gscale);
  DATA_SCALAR(ba);      // for Beta(a, b)
  DATA_SCALAR(bb);

  // ---- Parameters
  PARAMETER_VECTOR(theta);
  int n = Y.size();
  int d = Xhat.cols();

  // 1) unpack regression coefficients
  vector<Type> b    = theta.segment(0, d);

  // 2) unpack raw mixture‐log weights
  vector<Type> vraw = theta.segment(d, 3);
  vector<Type> expv = exp(vraw);
  Type sumv = expv.sum();

  // 3) mixture weights w00, w01, w10, w11
  vector<Type> w(4);
  w[0] = expv(0) / (Type(1) + sumv);
  w[1] = expv(1) / (Type(1) + sumv);
  w[2] = expv(2) / (Type(1) + sumv);
  w[3] = Type(1) - (w[0] + w[1] + w[2]);

  // 4) error standard deviations
  Type sigma0 = exp(theta(d+3));
  Type sigma1 = homoskedastic ? sigma0 : exp(theta(d+4));

  // 5) linear predictor
  vector<Type> mu = Xhat * b;

  // 6) build negative log‐likelihood
  Type nll = 0;
  for(int i = 0; i < n; i++){
    // mixture components for X=0 and X=1
    Type yi  = Y[i];
    Type mu_i= mu[i];

    // four log‐densities
    Type l1, l2, l3, l4;

    // -------------------------------
    // pick density by dist_code
    switch(dist_code){
    case 1: { // Normal
      l1 = dnorm(yi, mu_i,      sigma0, true);
      l2 = dnorm(yi, mu_i - b[0], sigma0, true);
      l3 = dnorm(yi, mu_i + b[0], sigma1, true);
      l4 = dnorm(yi, mu_i,       sigma1, true);
      break;
    }
    case 2: { // Student‐t, nu supplied
      Type z0 = (yi - mu_i) / sigma0;
      Type z1 = (yi - (mu_i - b[0])) / sigma0;
      Type z2 = (yi - (mu_i + b[0])) / sigma1;
      Type z3 = (yi - mu_i) / sigma1;
      // precompute constant
      Type tconst = lgamma((nu+1)/2) - lgamma(nu/2) - Type(0.5)*log(nu*M_PI);
      l1 = tconst - log(sigma0) - (nu+1)/2 * log(Type(1) + z0*z0/nu);
      l2 = tconst - log(sigma0) - (nu+1)/2 * log(Type(1) + z1*z1/nu);
      l3 = tconst - log(sigma1) - (nu+1)/2 * log(Type(1) + z2*z2/nu);
      l4 = tconst - log(sigma1) - (nu+1)/2 * log(Type(1) + z3*z3/nu);
      break;
    }
    case 3: { // Laplace
      l1 = -log(Type(2)*sigma0) - fabs(yi - mu_i)/sigma0;
      l2 = -log(Type(2)*sigma0) - fabs(yi - (mu_i - b[0]))/sigma0;
      l3 = -log(Type(2)*sigma1) - fabs(yi - (mu_i + b[0]))/sigma1;
      l4 = -log(Type(2)*sigma1) - fabs(yi - mu_i)/sigma1;
      break;
    }
    case 4: { // Gamma(shape=gshape, scale=gscale)
      l1 = dgamma(yi, gshape, gscale, true);
      l2 = l1;  l3 = l1;  l4 = l1;
      break;
    }
    case 5: { // Beta(alpha=ba, beta=bb)
      l1 = dbeta(yi, ba, bb, true);
      l2 = l1;  l3 = l1;  l4 = l1;
      break;
    }
    default:
      error("Unknown distribution code %d", dist_code);
    }
    // -------------------------------

    // exponentiate and weight
    Type term1_1 = w[3] * exp(l4);
    Type term2_1 = w[2] * exp(l2);
    Type term1_0 = w[1] * exp(l3);
    Type term2_0 = w[0] * exp(l1);

    if (Xhat(i,0) == Type(1)) {
      nll -= log(term1_1 + term2_1);
    } else {
      nll -= log(term1_0 + term2_0);
    }
  }

  return nll;
}
