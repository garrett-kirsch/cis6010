all: cugemm.exe cugemm-debug.exe cugemm-profile.exe

SOURCE_FILE=cugemm.cu

# optimized binary
cugemm.exe: $(SOURCE_FILE)
	nvcc -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# debug binary without optimizations
cugemm-debug.exe: $(SOURCE_FILE)
	nvcc -g -G -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# optimized binary with line number information for profiling
cugemm-profile.exe: $(SOURCE_FILE)
	nvcc -g --generate-line-info -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# NB: make sure you change the --algo flag here to profile the one you care about. 
# You can change the --export flag to set the filename of the profiling report that is produced.
profile: cugemm-profile.exe
	sudo /usr/local/cuda-11.8/bin/ncu --export my-profile --set full ./cugemm-profile.exe --size=1024 --reps=1 --algo=1 --validate=false

clean:
	rm -f cugemm*.exe
