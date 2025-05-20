// src/init.c
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Called automatically because you have useDynLib(MLBC, .registration=TRUE)
// in your NAMESPACE.
void R_init_MLBC(DllInfo *dll)
{
  // register zero routines—this satisfies the "must call R_registerRoutines" NOTE
  R_registerRoutines(dll, NULL, NULL, NULL, NULL);
  // but leave dynamic lookup on so TMB's own entry points
  // (getParameterOrder, etc.) remain visible to .Call()
  R_useDynamicSymbols(dll, TRUE);
}
