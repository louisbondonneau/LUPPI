/**********************************************
*version 1.01
*04/02/2020 stt_offs = 0.0;  j ai commente cette ligne car elle imposait un stt_offs de 0 a la place de seq_num*dt
*10/02/2020 quand on passe d'un timestamp pair à un tstamp impaire le bsn_diff = 195313 et tt_offs = (seq_num + 0.5)*tbin à la place de (seq_num - 0.5)*tbin
*10/02/2020 quand on passe d'un timestamp impair à un tstamp paire le bsn_diff = 195312 et tt_offs = (seq_num)*tbin à la place de (seq_num + 0.5)*tbin
*           
*********************************************/

#define _GNU_SOURCE
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <linux/if_ether.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <byteswap.h>
#include <math.h>
#include <sys/un.h>
#include <signal.h>
#include <omp.h>

#include "fitshead.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"
#include "mjdtime.h"
#include "cpu_utils.h"

#define STATUS "NETSTAT"
#include "threads.h"

void memsetOMP(char *mem, const short val, const long s, const long c)
{
    #pragma omp parallel for simd schedule(static,s/c)
    for(long i=s; i>0; --i)
        mem[i] = 0;
}

/* openmp memset acceleration */
void memsetopenmp(char *buf,  int X, size_t size)
{
    size_t my_start, my_size;
    
    if (omp_in_parallel())
    {
        int id = omp_get_thread_num();
        int num = omp_get_num_threads();
 
        my_start = (id*size)/num;
        my_size = ((id+1)*size)/num - my_start;
    }
    else
    {
        my_start = 0;
        my_size = size;
    }
 
    memset(buf + my_start, X, my_size);
}

/* Reset all counters */
void reset_stats(datablock_stats *d) {
    d->npacket=0;
    d->ndropped=0;
    d->last_pkt=0;
}

/* Reset block params */
void reset_block(datablock_stats *d) {
    d->block_idx = -1;
    d->packet_idx = 0UL;
    reset_stats(d);
}

/* Initialize block struct */
void init_block(datablock_stats *d, databuf *db, size_t packet_data_size, unsigned int packets_per_block, int overlap_packets) {
    d->db = db;
    d->packet_data_size = packet_data_size;
    d->packets_per_block = packets_per_block;
    d->overlap_packets = overlap_packets;
    reset_block(d);
}

/* Update block header info, set filled status */
void finalize_block(datablock_stats *d) {
    char *header = databuf_header(d->db, d->block_idx);
    hputi4(header, "PKTIDX", d->packet_idx);
    printf("debug PKTIDX = %d d->npacket = %d\n",d->packet_idx, d->npacket);
    hputi4(header, "PKTSIZE", d->packet_data_size);
    hputi4(header, "NPKT", d->npacket); // - d->overlap_packets); //ADD LOUIS - d->overlap_packets
    hputi4(header, "NDROP", d->ndropped);
    hputs (header, "DATATYPE", "RAW");  // GD
    //printf("packet_idx = %"PRIu64" block_idx=%d  packets_per_block=%d \n", d->packet_idx, d->block_idx, d->packets_per_block);
    databuf_set_filled(d->db, d->block_idx);
}

/* Push all blocks down a level, losing the first one */
void block_stack_push(datablock_stats *d, int nblock) {
    int i;
    for (i=1; i<nblock; i++) 
        memcpy(&d[i-1], &d[i], sizeof(datablock_stats));
}

/* Go to next block in set */
void increment_block(datablock_stats *d, uint64_t next_seq_num) {
    d->block_idx = (d->block_idx + 1) % d->db->n_block;
    d->packet_idx = next_seq_num - (next_seq_num % (d->packets_per_block - d->overlap_packets));
    //printf("packet_idx = %"PRIu64" block_idx=%d  packets_per_block=%d \n", d->packet_idx, d->block_idx, d->packets_per_block);
    reset_stats(d);
    // TODO: wait for block free here?
}

/* Check whether a certain seq num belongs in the data block */
int block_packet_check(datablock_stats *d, uint64_t seq_num) {
    if (seq_num < d->packet_idx) return(-1);
    else if (seq_num >= d->packet_idx + d->packets_per_block) return(1);
    else return(0);
}

/* Write a search mode (filterbank) style packet into the
 * datablock_stats.  Also zeroes out any dropped packets.
 */
void write_search_packet_to_block(datablock_stats *d, udp_packet *p) {
    const uint64_t seq_num = udp_packet_seq_num(p);
    int next_pos = seq_num - d->packet_idx;
    int cur_pos=0;
    if (d->last_pkt > d->packet_idx) cur_pos = d->last_pkt - d->packet_idx + 1;
    char *dataptr = databuf_data(d->db, d->block_idx) 
        + cur_pos*d->packet_data_size;
    for (; cur_pos<next_pos; cur_pos++) {
        memset(dataptr, 0, d->packet_data_size);
        dataptr += d->packet_data_size;
        d->npacket++;
        d->ndropped++;
    }
    udp_packet_data_copy(dataptr, p);
    d->last_pkt = seq_num;
    //d->packet_idx++; // XXX I think this is wrong..
    d->npacket++;
}

/* Write a baseband mode packet into the block.  Includes a 
 * corner-turn (aka transpose) of dimension nchan.
 */
void write_baseband_packet_to_block(datablock_stats *d, udp_packet *p, int nchan) {

    const uint64_t seq_num = udp_packet_seq_num(p);
    int block_pkt_idx = seq_num - d->packet_idx;
    udp_packet_data_copy_transpose( databuf_data(d->db, d->block_idx), nchan, block_pkt_idx, d->packets_per_block, p);

    /* Consider any skipped packets to have been dropped,
     * update counters.
     */
    if (d->last_pkt < d->packet_idx) d->last_pkt = d->packet_idx;

    if (seq_num == d->last_pkt) {
        d->npacket++;
    } else {
	//printf("d->npacket %d += seq_num %d - d->last_pkt %d\n",d->npacket , seq_num , d->last_pkt);
        d->npacket += seq_num - d->last_pkt;
        d->ndropped += seq_num - d->last_pkt - 1;
    }

    d->last_pkt = seq_num;
}



// -- Signal handlers for run control --
void handle_sigint(int sig) { run=0; }
void handle_sigusr1(int sig) { run=1; }




/* This thread is passed a single arg, pointer
 * to the udp_params struct.  This thread should 
 * be cancelled and restarted if any hardware params
 * change, as this potentially affects packet size, etc.
 */

void *net_thread(void *_args) {

  // -- Get arguments --
  thread_args *args = (thread_args *)_args;
  int first = 1;
  int paire = -1;
  int i;
  char strlog[128];

  // -- Install signal handlers --
  signal(SIGINT, cc);
  //signal(SIGUSR1, handle_sigusr1);

  // -- cpu affinity --
  printf("net_thread: setcpuaffinity(%d);\n",args->output_buffer - 1);
  setcpuaffinity(args->output_buffer - 1);

  // -- process priority --
  setabspriority(args->priority);

#if 0  
 setniceness(READNICENESS);
#endif
  // -- Time parameters -- 
  int stt_imjd=0, stt_smjd=0, smjd_start=0;
  double stt_offs=0.0;

  // -- Attach to status shared memory --
  int rv;
  status st;
  rv = status_attach(&st, args->gpu_id);
  if(rv!=OK) {
      log_error("net_thread","Error attaching to status shm");
      pthread_exit(NULL);
  }    
  pthread_cleanup_push((void *)status_detach, &st);
  pthread_cleanup_push((void *)set_exit_status, &st);
  log_info("net_thread","Attached to status shm OK");


  //  -- Read params from status shm and put it into temp status_buf --
  obs_params gp;
  struct psrfits pf;
  pf.sub.dat_freqs = NULL;
  pf.sub.dat_weights = NULL;
  pf.sub.dat_offsets = NULL;
  pf.sub.dat_scales = NULL;
  char status_buf[STATUS_SIZE];
  status_lock_safe(&st);
  memcpy(status_buf,st.buf,STATUS_SIZE);
  status_unlock_safe(&st);

  rv = read_status_params(status_buf, &gp, &pf);
  printf("read_status_params completed\n");
  if(rv!=OK) {
      log_error("net_thread","Error in read_status_params");
      exit(-1);
  }    
  log_info("net_thread","read_status_params completed");
  
  pthread_cleanup_push((void *)free_psrfits, &pf);

  // -- Init databuf struct --
  databuf *db=NULL;

  // -- Attach to databuf shared memory --
  db = databuf_attach(args->output_buffer);
  if(db == NULL) {
      sprintf(strlog,"Error attaching to data shm");
      log_error("net_thread",strlog );
      pthread_exit(NULL);
  }  
  pthread_cleanup_push((void *)databuf_detach, db);

  // -- Init struct socket --
  udp_params socket;
  // -- Read network params from status
  rv = read_net_params(status_buf, &socket);
  if(rv!=OK) {
      log_error("net_thread","Error in read_net_params");
      pthread_exit(NULL);
  }    
  log_info("net_thread","read_net_params completed");

  // -- Read forward network params from status
  int mode_forward;
  udp_params socket_fwd;
  uint64_t nbogus_forward_total;
  mode_forward = read_fwd_params(status_buf, &socket_fwd);
  sprintf(strlog,"GPU #%d Mode forward activated %d", args->gpu_id, mode_forward);
  log_info("net_thread",strlog);

  // -- Init struct packet --
  // Figure out size of data in each packet, number of packets per block, etc.
  udp_packet packet;
  int64_t block_size;
  printf("socket.packet_size= %d\n",socket.packet_size);
  size_t packet_data_size = udp_packet_datasize(socket.packet_size);
  printf("packet_data_size= %d\n",packet_data_size);
  unsigned packets_per_block;
  //db->block_size=4502515200;
  //hputi4(status_buf, "BLOCSIZE", 4502515200);
  

  hgeti5(status_buf, "BLOCSIZE", &block_size);
  printf("debug block_size = %lld\n",block_size);
  
  if (hgeti5(status_buf, "BLOCSIZE", &block_size)==0) {
      block_size = db->block_size;
      hputi5(status_buf, "BLOCSIZE", block_size);
  } else {
      printf("debug block_size = %lld\n",block_size);
      if (block_size > db->block_size) {
          sprintf(strlog,"BLOCSIZE > databuf block_size (%lld > %lld)", (int64_t)block_size, (int64_t)db->block_size);
          log_error("net_thread", strlog);
          printf("BLOCSIZE = %lld\n",block_size);
          block_size = db->block_size;
          hputi5(status_buf, "BLOCSIZE", block_size);
      }
  }
  printf("BLOCSIZE = %lld\n",block_size);fflush(stdout);
  packets_per_block = block_size / packet_data_size;
  printf("packet_data_size= %d\n",packet_data_size);
  printf("packets_per_block= %d\n",packets_per_block);

  // -- Set up socket --
  rv = udp_init(&socket);
  if(rv!=OK) {
      log_error("net_thread","error setting UDP connection");
      pthread_exit(NULL);
  } else {
      log_info("net_thread","init socket successfully");
  }
  pthread_cleanup_push((void *)udp_close, &socket);
  
  // -- Set up forward socket --
  if (mode_forward) {
      rv = udp_forward_init(&socket_fwd);
      if(rv!=OK) {
          log_error("net_thread","error setting UDP forward connection");
          pthread_exit(NULL);
      } else log_warn("net_thread","no UDP forward connection");  
  } else {
      log_info("net_thread","init socket forward successfully");
  }
  pthread_cleanup_push((void *)udp_close, &socket_fwd);

  // -- Set useful values --
  int nchan=0, npol=0;
  nchan = pf.hdr.nchan;
  npol = pf.hdr.npol;
  printf("nchan=%d npol=%d",nchan,npol);

  /* If we're in baseband mode, figure out how much to overlap
   * the data blocks.
   */
  int overlap_packets=0, baseband_packets=1;
  int samples_per_packet;
  if (baseband_packets) {
      if (hgeti4(status_buf, "OVERLAP", &overlap_packets)==0)
          overlap_packets = 0; // Default to no overlap
      else {
          // XXX This is only true for 8-bit, 2-pol data:
          samples_per_packet = packet_data_size / nchan / (size_t)4;
	  printf("samples_per_packet=%d  overlap_packets=%d\n",samples_per_packet,overlap_packets);
          if (overlap_packets % samples_per_packet) {
              log_error("net_thread", "Overlap is not an integer number of packets");
              overlap_packets = (overlap_packets/samples_per_packet+1);
              hputi4(status_buf, "OVERLAP", overlap_packets*samples_per_packet);
          } else {
              overlap_packets = overlap_packets/samples_per_packet;
	  }    
      }
  }
  sprintf(strlog, "samples_per_packet is %d", samples_per_packet);
  log_info("net_thread", strlog);
  sprintf(strlog, "packets_per_block is %d", packets_per_block);
  log_info("net_thread", strlog);
  sprintf(strlog, "Overlap packet is %d", overlap_packets);
  log_info("net_thread", strlog);
  sprintf(strlog, "Overlap is %d", overlap_packets*samples_per_packet);
  log_info("net_thread", strlog);

  // -- List of datablock_statss currently in use --
  const int nblock = 2;
  datablock_stats blocks[nblock];
  for (i=0; i<nblock; i++)
      init_block(&blocks[i], db, packet_data_size, packets_per_block, overlap_packets);

  // -- Convenience names for first/last blocks in set --
  datablock_stats *fblock, *lblock;
  fblock = &blocks[0];
  lblock = &blocks[nblock-1];

  /* Misc counters, etc */
  char *curdata=NULL, *curheader=NULL;
  int64_t seq_num_diff = 0UL;
  int64_t time_stamp_diff = 0UL;
  uint64_t seq_num = 0UL, last_seq_num=0UL, nextblock_seq_num=0UL, IP_id = 0UL, nbof_subpackets = 0UL, start_seq_num = 0UL;
  uint64_t time_stamp = 0UL, last_time_stamp=2048UL;
  double tbin;
  int one = 1;
  int printwait = 1;
  uint64_t npacket_total = 0UL, ndropped_total = 0UL, nbogus_total  = 0UL;
  ndropped_total = 0;
  double drop_frac_avg=0.0;
  const double drop_lpf = 0.25;

  // -- Timers var --
#ifdef STATS
  double t_total = 0.0;
  double t_cpytr = 0.0;
  double t_cpyfct = 0.0;
  double t_nextblock = 0.0;
  double t_udprcv = 0.0;
  double t_counter = 0.0;
  double t_udpwait = 0.0;
  struct timespec t_tot1, t_tot2, t_cpytr1, t_cpytr2,  t_cpyfct1, t_cpyfct2,  t_nextblock1, t_nextblock2;
  struct timespec t_udprcv1, t_udprcv2, t_counter1, t_counter2, t_udpwait1, t_udpwait2;
#endif


  /* Main loop */
  unsigned force_new_block=0, waiting=-1;

#ifdef STATS
  clock_gettime(CLOCK_REALTIME, &t_tot1);
#endif
  uint64_t cont=0;
  omp_set_num_threads(16);
  while(run) {

#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_udpwait1);
#endif
      rv = udp_wait(&socket);
      if (rv!=OK) {
          if (rv==TIMEOUT) {
              // -- Set "waiting" flag --
              if (waiting!=1) {
                  status_lock_safe(&st);
                  hputs(st.buf, STATUS, "waiting");
                  status_unlock_safe(&st);
                  waiting=1;
              }
	      log_info("net_thread","TIMEOUT");
              continue;
          } else {
              log_error("net_thread","udp_wait returned error");
	      pthread_exit(NULL);
          }
      }
#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_udpwait2);
      t_udpwait += (double)timediff(t_udpwait1, t_udpwait2); 
#endif

      // -- Read data from socket to packet --
#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_udprcv1);
#endif
      rv = udp_recv(&socket, &packet);
      if(rv != OK) {
          if (rv==ERR_PACKET) {
	      /* Unexpected packet size, ignore? */
	      nbogus_total++;
	      continue;
          } else {
              log_error("net_thread","Error within udp_recv");
	      pthread_exit(NULL);
          }
      }

      // -- Forward the packet --
      if (mode_forward) {
	  rv = udp_forward(&socket_fwd, &packet);
	  if(rv != OK) {
	      if (rv==ERR_PACKET) {
		  /* Unexpected packet size, ignore? */
		  nbogus_forward_total++;
		  continue;
	      } else {
		  log_error("net_thread","Error within udp_forward");
		  pthread_exit(NULL);
	      }
	  }
      }

#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_udprcv2);
      t_udprcv += (double)timediff(t_udprcv1, t_udprcv2); 
#endif


      // -- Update status if needed --
      if (waiting!=0) {
          status_lock_safe(&st);
          hputs(st.buf, STATUS, "receiving");
          status_unlock_safe(&st);
          waiting=0;
      }

      // -- Check counter packet --
#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_counter1);
#endif

      IP_id = udp_packet_IP_id(&packet);
      nbof_subpackets = udp_nbof_subpackets(&packet);


      seq_num = udp_packet_seq_num(&packet);
      time_stamp = udp_packet_time_stamp(&packet);
      


      //if( (first == 1) && (one == 1) ) {
      //    one = 2;
      //    goto initialization;
      //}
      if( (first == 1) && (one == 1) ) {
          last_seq_num = seq_num - nbof_subpackets;
          last_time_stamp = time_stamp;
          one=0;
      }

      seq_num_diff = (int64_t) ((uint64_t)seq_num - (uint64_t)last_seq_num);
      time_stamp_diff = time_stamp - last_time_stamp;


      force_new_block=0;

      if (first == 1) {
          if ( (int64_t)time_stamp_diff == (int64_t)1) {
              stt_imjd = 40587 + time_stamp/86400;
              stt_smjd = time_stamp%86400;
              get_int(status_buf,"MJDSTART", &smjd_start, 1);
              get_dbl(status_buf, "TBIN", &tbin, 30.0);
              if(printwait) {
                  printf("waiting for smjd=%d curently %d\n", smjd_start, stt_smjd);
                  sprintf(strlog, "waiting for smjd=%d curently %d", smjd_start, stt_smjd);
                  log_info("net_thread", strlog);
                  printwait = 0;
              }
              printf("waiting for smjd=%d curently %d\n", smjd_start, stt_smjd);
              if((stt_smjd > smjd_start) && (stt_smjd-smjd_start < 600 ) ) {
                  printf("Force start smjd=%d curently %d\n", smjd_start, stt_smjd);
                  sprintf(strlog, "Force start smjd=%d curently %d", smjd_start, stt_smjd);
                  log_info("net_thread", strlog);
                  smjd_start = -1;
              }
              if( ((int64_t)stt_smjd == (int64_t)smjd_start) || (smjd_start < 0) ) {
                  if ( seq_num_diff+195312 == nbof_subpackets ) {
                      seq_num_diff += 195312;
                      printf("second packet impaire seq num = %d\n",seq_num);
                      stt_offs = (seq_num)*tbin;
                      sprintf(strlog, "imjd=%d smjd=%d offs=%lf", stt_imjd, stt_smjd, stt_offs);
                      log_info("net_thread", strlog);
                      paire=0;
                      force_new_block=1;
                      start_seq_num = seq_num;
                      first = 0;
                      cont += 1;
                      npacket_total += 1;
                      ndropped_total = 0;
                  }
                  else if ( seq_num_diff+195313 == nbof_subpackets ) {
                      seq_num_diff += 195313;
                      printf("second packet paire seq num = %d\n",seq_num);
                      stt_offs = (seq_num + 0.5)*tbin;
                      sprintf(strlog, "imjd=%d smjd=%d offs=%lf", stt_imjd, stt_smjd, stt_offs);
                      log_info("net_thread", strlog);
                      paire=1;
                      force_new_block=1;
                      start_seq_num = seq_num;
                      first = 0;
                      cont += 1;
                      npacket_total += 1;
                      ndropped_total = 0;
                  }
                  else {
                      sprintf(strlog, "3Received out-of-order packet (seq_num=%"PRIu64", last_seq_num=%"PRIu64", diff=%"PRId64"  time_stamp_diff=%"PRIu64")", seq_num, last_seq_num, seq_num_diff, time_stamp_diff);
                      log_warn("net_thread", strlog);
                      npacket_total += (int)((float)seq_num_diff/(float)nbof_subpackets);
                      ndropped_total += (int)((float)seq_num_diff/(float)nbof_subpackets) - 1;
                      cont += (int)((float)seq_num_diff/(float)nbof_subpackets);
                  }
              }
          }
      }
      else { //if (first == 0) {
            if ( ((int64_t)time_stamp_diff == (int64_t)0) ) { // dts = 0 & first == 0
                  if ( (int64_t)seq_num_diff == (int64_t)nbof_subpackets ) {
                      // -- No going backwards --
                      cont += 1;
                      npacket_total += 1;
                  }
                  else {
                      sprintf(strlog, "0Received out-of-order packet (seq_num=%"PRIu64", last_seq_num=%"PRIu64", diff=%"PRIu64"  time_stamp_diff=%"PRIu64")", seq_num, last_seq_num, seq_num_diff, time_stamp_diff);
                      cont += (int)((float)seq_num_diff/nbof_subpackets);
                      log_warn("net_thread", strlog);
                      npacket_total += (int)((float)seq_num_diff/(float)nbof_subpackets);
                      ndropped_total += (int)((float)seq_num_diff/(float)nbof_subpackets) - 1;
                  }
            }
            else if ( (int64_t)time_stamp_diff == (int64_t)1) { // dts = 1 & first == 0
                //sprintf(strlog, "Received time_stamp_diff cont = %d", cont);
                //log_info("net_thread", strlog);
                if ( paire == 1 ) {
                    printf("paire 1 nbof_subpackets-seq_num_diff = %"PRIu64"\n", (nbof_subpackets-seq_num_diff));
                    if ( seq_num_diff+195312 == nbof_subpackets ) {
                        cont += 1;
                        npacket_total += 1;
                        seq_num_diff += 195312;
                    }
                    else {
                        seq_num_diff += 195312;
                        sprintf(strlog, "1Received out-of-order packet (seq_num=%"PRIu64", last_seq_num=%"PRIu64", diff=%"PRIu64"  time_stamp_diff=%"PRIu64")", seq_num, last_seq_num, seq_num_diff, time_stamp_diff);
                        cont += (int)((float)seq_num_diff/nbof_subpackets);
                        log_warn("net_thread", strlog);
                        npacket_total += (int)((float)seq_num_diff/(float)nbof_subpackets);
                        ndropped_total += (int)((float)seq_num_diff/(float)nbof_subpackets) - 1;
                    }
                    paire=0;
                }
                else{ // paire ==0
                    printf("paire 0 nbof_subpackets-seq_num_diff = %"PRIu64"\n", (nbof_subpackets-seq_num_diff));
                    if ( seq_num_diff+195313 == nbof_subpackets ) {
                        cont += 1;
                        npacket_total += 1;
                        seq_num_diff += 195313;
                    }
                    else {
                        seq_num_diff += 195313;
                        sprintf(strlog, "2Received out-of-order packet (seq_num=%"PRIu64", last_seq_num=%"PRIu64", diff=%"PRIu64"  time_stamp_diff=%"PRIu64")", seq_num, last_seq_num, seq_num_diff, time_stamp_diff);
                        log_warn("net_thread", strlog);
                        npacket_total += (int)((float)seq_num_diff/(float)nbof_subpackets);
                        ndropped_total += (int)((float)seq_num_diff/(float)nbof_subpackets) - 1;
                        cont += (int)((float)seq_num_diff/nbof_subpackets);
                    }
                    paire=1;
                }
            }
      }

      last_seq_num = seq_num;
      last_time_stamp = time_stamp;
      //initialization:
      
      /*if ( cont == 112000 ) {
      		sprintf(strlog, "intentionally dropped packet for 2 sec (seq_num=%"PRIu64", last_seq_num=%"PRIu64", diff=%"PRIu64"  cont=%"PRIu64")", seq_num, last_seq_num, seq_num_diff, cont);
      		log_info("net_thread", strlog);
		sleep(2);
      }*/


      (*(uint32_t *)((packet.data) + 12)) = (uint32_t)cont;  //udp_packet_seq_num(p) --> cont
      seq_num = (uint64_t)cont;


#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_counter2);
      t_counter += (double)timediff(t_counter1, t_counter2); 
#endif


      /* Determine if we go to next block */
#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_nextblock1);
#endif
      //printf("seq_num =%d >= nextblock_seq_num = %d\n",seq_num, nextblock_seq_num);
      if ((seq_num >= nextblock_seq_num) || force_new_block) {
	  //printf("XXXXXXXXXXXXXXXXXXXXXX   seq_num=%"PRIu64", nextblock_seq_num=%"PRIu64"\n", seq_num, nextblock_seq_num);
          
          /* Update drop stats */
	  //printf("fblock->npacket= %d\n",fblock->npacket);
	  //printf("fblock->packet_idx= %d\n",fblock->packet_idx);
          if (fblock->npacket)  
              drop_frac_avg = (1.0-drop_lpf)*drop_frac_avg + drop_lpf * (double)fblock->ndropped / (double)fblock->npacket;

	  //printf("fblock->block_idx = %d\n",fblock->block_idx);
          //status_lock_safe(&st);
          //hputr8(st.buf, "DROPAVG", drop_frac_avg);
          //hputr8(st.buf, "DROPTOT", (double) npacket_total ? (double)ndropped_total/(double)npacket_total : 0.0);
          //hputr8(st.buf, "DROPBLK", (double) fblock->npacket ? (double)fblock->ndropped/(double)fblock->npacket : 0.0);
          //status_unlock_safe(&st);

          /* Finalize first block, and push it off the list.
           * Then grab next available block.
           */
          if (fblock->block_idx>=0) finalize_block(fblock);



          block_stack_push(blocks, nblock);
          increment_block(lblock, seq_num);
          curdata = databuf_data(db, lblock->block_idx);
          curheader = databuf_header(db, lblock->block_idx);
	  //printf("lblock->packet_idx = %d  packets_per_block= %d overlap_packets= %d\n",lblock->packet_idx,packets_per_block,overlap_packets);
          nextblock_seq_num = lblock->packet_idx + packets_per_block - overlap_packets;

          /* If new obs started, reset total counters, get start
           * time.  Start time is rounded to nearest integer
           * second, with warning if we're off that by more
           * than 100ms.  Any current blocks on the stack
           * are also finalized/reset */

          //printf("force_new_block =%d \n",force_new_block);fflush(stdout);
          if (force_new_block) {
              /* Reset stats */
              printf("force new block \n");fflush(stdout);
              npacket_total=0UL;
              ndropped_total=0UL;
              nbogus_total=0UL;
              cont=0;

              /* Get obs start time */
	      //printf("Get current MJD\n");fflush(stdout);
              //get_current_mjd(&stt_imjd, &stt_smjd, &stt_offs);
	      sprintf(strlog, "imjd=%d smjd=%d offs=%lf", stt_imjd, stt_smjd, stt_offs);
	      log_info("net_thread", strlog);
              if (stt_offs>0.5) { stt_smjd+=1; stt_offs-=1.0; }
              if (fabs(stt_offs)>0.1) { 
                  sprintf(strlog, "Second fraction = %3.1f ms > +/-100 ms", stt_offs*1e3);
                  log_info("net_thread", strlog);
              }
              //stt_offs = 0.0;   //04/02/2020 louis

              /* Warn if 1st packet number is not zero */
              if (seq_num!=0) {
                  sprintf(strlog, "First packet number is not 0 (seq_num=%"PRIu64")", start_seq_num);
                  log_warn("net_thread", strlog);
              }

              /* Flush any current buffers */
              for (i=0; i<nblock-1; i++) {
                  if (blocks[i].block_idx>=0) finalize_block(&blocks[i]);
                  reset_block(&blocks[i]);
              }

          }

          /* Read/update current status shared mem */
          status_lock_safe(&st);
          hputr8(st.buf, "DROPAVG", drop_frac_avg);
          hputr8(st.buf, "DROPTOT", (double) npacket_total ? (double)ndropped_total/(double)npacket_total : 0.0);
          hputr8(st.buf, "DROPBLK", (double) fblock->npacket ? (double)fblock->ndropped/(double)fblock->npacket : 0.0);
          if (stt_imjd!=0) {
              hputi4(st.buf, "STT_IMJD", stt_imjd);
              hputi4(st.buf, "STT_SMJD", stt_smjd);
              hputr8(st.buf, "STT_OFFS", stt_offs);
              hputi4(st.buf, "STTVALID", 1);
          } else {
              hputi4(st.buf, "STTVALID", 0);
          }
          memcpy(status_buf, st.buf, STATUS_SIZE);
          status_unlock_safe(&st);

          /* block size possibly changed on new obs */
          /* TODO: what about overlap...
           * Also, should this even be allowed ?
           */
          if (force_new_block) {
              if (hgeti5(status_buf, "BLOCSIZE", &block_size)==0) {
                      block_size = db->block_size;
              } else {
                  if (block_size > db->block_size) {
                      log_error("net_thread", "BLOCSIZE > databuf block_size");
                      block_size = db->block_size;
                  }
              }
              packets_per_block = block_size / packet_data_size;
          }
          //hputi5(status_buf, "BLOCSIZE", block_size);

          /* Wait for new block to be free, then clear it
           * if necessary and fill its header with new values.
           */
          hgeti5(status_buf, "BLOCSIZE", &block_size);
          printf("net_thread: BLOCSIZE = %lld\n",block_size);
          printf("block %u\n",lblock->block_idx);
          while ((rv=databuf_wait_free(db, lblock->block_idx)) != OK) {
              if (rv==TIMEOUT) {
                  printf("Wait on block %u to be free\n",lblock->block_idx);
                  log_info("net_thread", "Wait for new block to be free");
                  waiting=1;
                  status_lock_safe(&st);
                  hputs(st.buf, STATUS, "blocked");
                  status_unlock_safe(&st);
                  continue;
              } else {
                  log_error("net_thread", "error waiting for free databuf");
                  run=0;
                  pthread_exit(NULL);
                  break;
              }
          }
          memcpy(curheader, status_buf, STATUS_SIZE);
          //double tmr;
          //tmr = -omp_get_wtime();
          //#pragma omp parallel
          //{
          //if (1) { memsetOMP(curdata, 0, (size_t)block_size,16); }
          //}
          //tmr += omp_get_wtime();
          //printf("memsetopenmp:   %.3f MB/s\n", block_size/(1.e+6*tmr));
      }
#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_nextblock2);
      t_nextblock += (double)timediff(t_nextblock1, t_nextblock2); 
#endif


      /* Copy packet into any blocks where it belongs.
       * The "write packets" functions also update drop stats 
       * for blocks, etc.
       */
#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_cpyfct1);
#endif
      for (i=0; i<nblock; i++) {
          if ((blocks[i].block_idx>=0) && (block_packet_check(&blocks[i],seq_num)==0)) {
              if (baseband_packets) {
                  write_baseband_packet_to_block(&blocks[i], &packet, nchan);
	      }	  
              else
                  write_search_packet_to_block(&blocks[i], &packet);
          }
      } 

#ifdef STATS
      clock_gettime(CLOCK_REALTIME, &t_cpyfct2);
      t_cpyfct += (double)timediff(t_cpyfct1, t_cpyfct2); 
#endif

      // -- Will exit if thread has been cancelled -- 
      pthread_testcancel();
  }

#ifdef STATS
  clock_gettime(CLOCK_REALTIME, &t_tot2);
  t_total = (double)timediff(t_tot1, t_tot2);

  printf("Total time        :  %12.8f s\n", t_total);
  printf("UDP wait          :  %12.8f s  (%05.2f%%)\n", t_udpwait, 100 * t_udpwait/t_total);
  printf("UDP receive       :  %12.8f s  (%05.2f%%)\n", t_udprcv, 100 * t_udprcv/t_total);
  printf("Counter fct       :  %12.8f s  (%05.2f%%)\n", t_counter, 100 * t_counter/t_total);
  printf("Copy call         :  %12.8f s  (%05.2f%%)\n", t_cpyfct, 100 * t_cpyfct/t_total);
  printf("Next blocks       :  %12.8f s  (%05.2f%%)\n", t_nextblock, 100 * t_nextblock/t_total);
  fflush(stdout);
#endif
  pthread_exit(NULL);

  // -- Have to close all push's -- 
  pthread_cleanup_pop(0); /* Closes push(udp_close) */
  pthread_cleanup_pop(0); /* Closes set_exit_status */
  pthread_cleanup_pop(0); /* Closes free_psrfits */
  pthread_cleanup_pop(0); /* Closes status_detach */
  pthread_cleanup_pop(0); /* Closes databuf_detach */
  pthread_cleanup_pop(0); /* Closes push(udp_close) forward */

}

