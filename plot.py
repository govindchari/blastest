from matplotlib import pyplot as plt

gemv_results = "gemv_results.csv"

# Read the file content
with open(gemv_results, 'r') as file:
    custom_gflops = file.readline().strip().split(',')
    blas_gflops = file.readline().strip().split(',')
custom_gflops = [float(item.strip()) for item in custom_gflops[0:len(custom_gflops)-1]]
blas_gflops = [float(item.strip()) for item in blas_gflops[0:len(blas_gflops)-1]]
N = len(custom_gflops)

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
Nlist = [2**(x + 1) for x in range(N)]
plt.figure()
plt.plot(Nlist, custom_gflops, "-o", label="Custom BLAS", color="mediumseagreen")
plt.plot(Nlist, blas_gflops, "-o", label="OpenBLAS",  color="darkviolet")
plt.xscale("log")

plt.xlabel("Rows of Matrix")
plt.ylabel("GFLOPs")
plt.title("GFLOPs vs Matrix Size for gemv")
plt.legend()
plt.savefig("gemv_GFLOPS.pdf", bbox_inches='tight')
plt.show()
