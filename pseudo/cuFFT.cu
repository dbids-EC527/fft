cufftHandle plan;
cufftPlan2d(&plan, rowLen, rowLen, CUFFT_Z2Z);
cufftExecZ2Z(plan, d_array, d_array, CUFFT_FORWARD) != CUFFT_SUCCESS);