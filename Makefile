NVCC = nvcc

SOURCEDIR = src
SOURCECOMMONSDIR = src/commons
BINDIR = bin

XFLAGS = -fopenmp -DUSE_NVTX
NVCCFLAGS = -arch=native -m64 -O3 -rdc=true -lnvToolsExt

CONFIG = config
TARGETS = main

all : clean build

build : $(TARGETS)

clean :
	rm -rf $(BINDIR)

$(BINDIR)/%.bin : $(SOURCEDIR)/$(CONFIG).cuh $(SOURCEDIR)/%.cu $(BINDIR)/functions.o
	$(NVCC) -Xcompiler="$(XFLAGS)" $(NVCCFLAGS) $(BINDIR)/functions.o $(SOURCEDIR)/$*.cu -o $(BINDIR)/$*.bin

$(TARGETS) : % : $(BINDIR)/%.bin

$(BINDIR)/functions.o : $(SOURCECOMMONSDIR)/functions.cu $(SOURCECOMMONSDIR)/functions.cuh
	mkdir -p $(BINDIR)
	$(NVCC) -c $(NVCCFLAGS) $(SOURCECOMMONSDIR)/functions.cu -o $(BINDIR)/functions.o
