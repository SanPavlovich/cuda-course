compile:
	nvcc gemm_naive.cu -o gemm_naive -lnvToolsExt
profile:
	ncu --set full -f --nvtx -o gemm_naive gemm_naive