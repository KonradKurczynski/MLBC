// src/likelihood_unlabeled.cpp
#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() () {
  // ---- Data from R
  DATA_VECTOR(Y);
  DATA_MATRIX(Xhat);
  DATA_INTEGER(homoskedastic);

  // ---- Parameters
  PARAMETER_VECTOR(theta);
  int n = Y.size();
  int d = Xhat.cols();

  // 1) unpack b
  vector<Type> b    = theta.segment(0, d);
  // 2) unpack raw mixture logs
  vector<Type> vraw = theta.segment(d, 3);
  vector<Type> expv = exp(vraw);
  Type sumv = expv.sum();
  // 3) build weights w00,w01,w10,w11
  vector<Type> w(4);
  w[0] = expv(0) / (Type(1) + sumv);
  w[1] = expv(1) / (Type(1) + sumv);
  w[2] = expv(2) / (Type(1) + sumv);
  w[3] = Type(1) - (w[0] + w[1] + w[2]);

  // 4) sigmas
  Type sigma0 = exp(theta(d+3));
  Type sigma1 = homoskedastic ? sigma0 : exp(theta(d+4));

  // 5) linear predictor
  vector<Type> mu = Xhat * b;

  // 6) negative log‑likelihood
  Type nll = 0;
  for(int i = 0; i < n; i++){
    // get the two mixture densities (in log‑space, then exponentiate)
    Type log1 = dnorm(Y[i], mu[i],      sigma0, true);
    Type log2 = dnorm(Y[i], mu[i] - b[0], sigma0, true);
    Type log3 = dnorm(Y[i], mu[i] + b[0], sigma1, true);
    Type log4 = dnorm(Y[i], mu[i],       sigma1, true);

    Type term1_1 = w[3] * exp(log4);
    Type term2_1 = w[2] * exp(log2);
    Type term1_0 = w[1] * exp(log3);
    Type term2_0 = w[0] * exp(log1);

    if (Xhat(i,0) == Type(1)) {
      nll -= log(term1_1 + term2_1);
    } else {
      nll -= log(term1_0 + term2_0);
    }
  }

  return nll;
}
