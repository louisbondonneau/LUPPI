CUDA = /usr/local/cuda
PRESTO = /home/louis/Pulsar/presto
#FORWARD = -DHAVE_FORWARD
FORWARD = 
DEBUG_STATS = -DSTATS
DEBUG_STATS = 
CFLAGS = $(FORWARD) $(DEBUG_STATS) -O3 -mtune=native -march=native -Wall -I$(CUDA)/include -I/usr/lib/x86_64-linux-gnu
NVCCFLAGS = -gencode arch=compute_61,code=sm_61 -I$(CUDA)/include -I/usr/lib/x86_64-linux-gnu -Xcompiler -fopenmp
INSTALL_DIR = /home/louis/luppi

#PROGS = check_databuf check_status clean_shmem \
	test_udp_recv test_psrfits test_psrfits_read fold_psrfits \
	fix_psrfits_polyco psrfits_singlepulse
PROGS = check_raw_data
OBJS  = status.o databuf.o udp.o logging.o \
	params.o mjdtime.o thread_args.o \
	write_psrfits.o read_psrfits.o misc_utils.o \
	fold.o polyco.o hget.o hput.o sla.o downsample.o \
	dedisperse_utils.o cpu_utils.o

THREAD_PROGS = luppi_daq luppi_daq_ds luppi_daq_dedisp_GPU1 luppi_write_raw luppi_read_raw getmjdtime
THREAD_OBJS  = net_thread.o rawdisk_thread.o psrfits_thread.o fold_thread.o null_thread.o disk2mem_thread.o

CUDA_OBJS = dedisp_thread.o dedisp_ds_thread.o ds_thread.o fold_gpu.o dedisperse_gpu.o downsample_gpu.o stats_gpu.o cuda_utils.o
LIBS = -lcfitsio -L$(PRESTO)/lib -lsla -lm -lpthread -lrt -lstdc++ -lgomp
CUDA_LIBS = -L$(CUDA)/lib64 -lcufft -lcuda -lcudart -lrt -lm

all: $(PROGS) $(THREAD_PROGS) 
clean:
	rm -f $(PROGS) $(THREAD_PROGS) psrfits.tgz *~ *.o test_psrfits_0*.fits *.ptx

install: $(PROGS) $(THREAD_PROGS) 
	mkdir -p $(INSTALL_DIR) && \
	cp -f $(PROGS) $(THREAD_PROGS) psrfits_subband test_dedisp_thread luppi_daq_dedisp luppi_daq_dedisp_GPU1 luppi_daq_dedisp_GPU2 $(INSTALL_DIR)

psrfits.tgz: psrfits.h read_psrfits.c write_psrfits.c polyco.c polyco.h \
	PSRFITS_v3.4_fold_template.txt \
	PSRFITS_v3.4_search_template.txt
	tar cvzf $@ $^

find_dropped_blocks: find_dropped_blocks.o 
	$(CC) $(CFLAGS) $< -o $@ -lcfitsio -lm

psrfits_subband: psrfits_subband.c psrfits_subband_cmd.o $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ psrfits_subband_cmd.o $(OBJS) $(LIBS)

%.o : %.cu
	nvcc -c $(NVCCFLAGS) $< -o $@

test_dedisp_thread: test_dedisp_thread.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

luppi_daq_dedisp: luppi_daq_dedisp.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)
luppi_daq_dedisp_GPU1: luppi_daq_dedisp_GPU1.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)
luppi_daq_dedisp_GPU2: luppi_daq_dedisp_GPU2.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

luppi_daq_ds: luppi_daq_ds.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)


luppi_daq: luppi_daq.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

luppi_daq: luppi_daq.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

luppi_write_raw: luppi_write_raw.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

luppi_read_raw: luppi_read_raw.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

getmjdtime: getmjdtime.c $(OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@  $(OBJS) $(LIBS)

.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(OBJS) $(LIBS) $(THREAD_LIBS)
