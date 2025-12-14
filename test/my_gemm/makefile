compile_c:
	gcc gemm.c -o gemmc
compile_cu:
	nvcc gemm.cu -o gemmcu
run_cu:
	./gemmcu
profile_cu:
	ncu -o profile --set full --target-processes all -f ./gemmcu
