#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <poll.h>
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
#include <sys/time.h>

#include "CEP_selection_ring.h"
#include "logging.h"
#include "databuf.h"
#include "net_thread.h"
#include "mydates.h"

//#include "threads.h"

// A VERIFIER !
#define MAX_PACKET_SIZE 10000

// -------------------------------------------------------------------------------------------
void *net_thread(void *args)
{int i=0,k,ib=0,ibt=0,rv,run,ichan,isamp,bytes_per_sample,cpuaff,lastdropblk=0,currentdroppedsamp;
char prgid[32],strlog[256],string[256];
int databuf_id,old_cur_buf;
int sock;	// Receive socket
int rcv_port;	// Receive port
//unsigned char *packet;
CEP_frame_t *packet;
//BUF_frame_t *packetbuffer;
unsigned char *ptr;
int nofblocks,nobpb,bm,packetsize,headersize=16,offset=0;
int nofblocksc,nobpbc,bmc;
uint64_t ts,bsn,lastts=0,lastbsn=0,diffts,currenttime2ts,tsc,bsnc,nbbsn;
int64_t diffbsn;
int startoneven= 1;
struct tm *datetime; struct timeval tv;
long int possiblediffbsn1,possiblediffbsn2,possiblediffbsn3;
char *data,*oldata,*ZEROdata,smalldata[16];
char *iptr, *optr;  char *iptr_base, *optr_base;
char *optr2;
char hostname[256];
char dst_ip[256];
char rcv_ip[256];
struct ip_mreq group;
char simulated[1024];	// to host simulated data (for one channel)
uint64_t effective_idx_samp=0; int current_effective_idx_samp=0;
int max_samp_per_buffer,samp_per_buffer,nfftc,fftlenc,nfft2intc,overlap=0,offset_idx_samp=0,sizeoverlap=0;
float xr,xi,yr,yi, f;
double dp,df,time2flush,losttime;  char date[64],heure[64];
FILE *pfo;
int flgsimulated=0;

//offset=42;	// for test in prismatic !!
databuf_id = *((int *) args);

gethostname(hostname,256);
sprintf(prgid,"net_thread_%01d",databuf_id);
sprintf(strlog,"Starting net_thread  (databuf_id= %d)",databuf_id+1); log_info(prgid,strlog); myprintf(prgid,strlog);

// -- cpu affinity --
cpuaff = 4 * (databuf_id - 1) + 0;
cpuaff = 4 * databuf_id  + 0;
cpuaff = 16 * databuf_id  + 0;
setcpuaffinity( cpuaff );
sprintf(strlog,"set cpuaffinity to %d",cpuaff); log_info(prgid,strlog);
// -process priority
setabspriority(15);

// -- Create ZEROdata buf
if((ZEROdata= (char *)malloc(DATABUF_SIZE*1024*1024)) == NULL) {
  sprintf(strlog,"Unable to ZEROdata=malloc(%d) !",DATABUF_SIZE*1024*1024); log_info(prgid,strlog); myprintf(prgid,strlog);
} else {
  memset(ZEROdata, 0, DATABUF_SIZE*1024*1024);
  sprintf(strlog,"created a ZEROdata buffer full of zero to initialize databuf quickly (%dbytes)",DATABUF_SIZE*1024*1024); log_info(prgid,strlog);
}

// Init databuf struct
databuf *db=NULL;

//  Attach to databuf shared memory
db = databuf_attach(databuf_id+1);
if(db == NULL) {
  sprintf(strlog,"Error attaching to data shm");
  log_error(prgid,strlog );
  pthread_exit(NULL);
}
sprintf(strlog,"databuf_attach OK  (databuf_id= %d shmid= %d %p)",databuf_id+1,db->shmid,db); log_info(prgid,strlog); myprintf(prgid,strlog);
sprintf(strlog," pointer to .fftlen %p  fftlen= %d",&(db->fftlen),db->fftlen); log_info(prgid,strlog);
db->cur_buf= 0;
db->dropblk=0;
db->eis=0;
db->run=0;
for(i=0;i<db->n_block;i++) { db->eisb[i]=0; db->tsb[i]=0; db->bsnb[i]=0; }
iptr = databuf_data(db,0);
//for(i=0;i<db->n_block;i++) {
//  data = databuf_data(db,i);
//  sprintf(strlog,"databuf block %d at %p (diff= %ld)",i,data,(long int)(data-iptr)); log_info(prgid,strlog);
//  iptr=data;
//}
// memset(data,0,DATABUF_SIZE*1024*1024); is VERY SLOW ~45/50ms... should try memcpy from a ZEROed buffer
sprintf(strlog,"memcpy ZEROdata to all %d data db",db->n_block); log_info(prgid,strlog);
for(i=0;i<db->n_block;i++) {
  data = databuf_data(db,i);
  //sprintf(strlog,"memcpy (data_%d= %p, ZEROdata)",i,data); myprintf(prgid,strlog); log_info(prgid,strlog);
  memcpy(data, ZEROdata, DATABUF_SIZE*1024*1024);
  //sprintf(strlog,"memcpy OK"); myprintf(prgid,strlog);
}
log_info(prgid," ... memcpy ZEROdata done!");
data = databuf_data(db,db->cur_buf);

// UDP packet memory allocation
packet = malloc(sizeof(CEP_frame_t));
sprintf(strlog,"allocating packet with %lu bytes",sizeof(CEP_frame_t));
log_info(prgid,strlog);

// Set up socket
// Init struct socket
sprintf(strlog,"open UDP socket on lane %d",db->currentlane); log_info(prgid,strlog);
switch (db->currentlane){
  case 0:
    rcv_port = CEP_TX_UDPPORT_0;
    snprintf(dst_ip, 256, CEP_TX_IP_0);
    snprintf(rcv_ip, 256, UNDYS_RX_IP_0);
    break;
  case 1:
    rcv_port = CEP_TX_UDPPORT_1;
    snprintf(dst_ip, 256, CEP_TX_IP_1);
    snprintf(rcv_ip, 256, UNDYS_RX_IP_1);
    break;
  case 2:
    rcv_port = CEP_TX_UDPPORT_2;
    snprintf(dst_ip, 256, CEP_TX_IP_2);
    snprintf(rcv_ip, 256, UNDYS_RX_IP_2);
    break;
  case 3:
    rcv_port = CEP_TX_UDPPORT_3;
    snprintf(dst_ip, 256, CEP_TX_IP_3);
    snprintf(rcv_ip, 256, UNDYS_RX_IP_3);
    break;
  default:
    sprintf(strlog,"Lane out of range: %d", db->currentlane); log_error(prgid,strlog );
    pthread_exit(NULL);
    break;
}

sprintf(strlog,"DST=%s:%d RCV=%s", dst_ip, rcv_port, rcv_ip); log_info(prgid,strlog);



sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
if (sock==-1) {
  sprintf(strlog,"Error socket()"); log_error(prgid,strlog );
  pthread_exit(NULL);
}

sprintf(strlog,"Overwriting port from locale.h: %d", rcv_port); log_info(prgid,strlog);
rcv_port = db->UDPport;
if(db->UDPport == 0) {
  db->UDPport= UDP_PORT;
  rcv_port=UDP_PORT;
}
sprintf(strlog,"UDP port is now %d", rcv_port); log_info(prgid,strlog);


sprintf(strlog,"Will try to bind to UDP packets on port %d",rcv_port); log_info(prgid,strlog);
// bind to local address
struct sockaddr_in local_ip;
local_ip.sin_family =  AF_INET;
local_ip.sin_port = htons(rcv_port);
local_ip.sin_addr.s_addr = inet_addr(rcv_ip);
rv = bind(sock, (struct sockaddr *)&local_ip, sizeof(local_ip));
if (rv==-1) {
  sprintf(strlog,"Error bind()"); log_error(prgid,strlog );
  db->run= 0;
  pthread_exit(NULL);
}


// Non-blocking recv
fcntl(sock, F_SETFL, O_NONBLOCK);
sprintf(strlog,"  socket conf : fcntl(sock, F_SETFL, O_NONBLOCK)"); log_info(prgid,strlog);



if (strncmp(dst_ip,"224.",4) == 0){
  sprintf(strlog,"Will try to join Multicast Group %s", dst_ip); log_info(prgid,strlog);
  group.imr_multiaddr.s_addr = inet_addr(dst_ip);
  group.imr_interface.s_addr = inet_addr(rcv_ip);
  if(setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&group, sizeof(group)) < 0){
    sprintf(strlog,"Adding multicast group error"); log_error(prgid,strlog );
    close(sock);
    pthread_exit(NULL);
  }
}


//printf("sizeof(CEP_frame_t)= %d\n",sizeof(CEP_frame_t));

// 'poll' setup
struct pollfd pfd;
pfd.fd = sock;
pfd.events = POLLIN;

// Got the FIRST PACKET to determine different size
log_info(prgid,"waiting for special first packet to get info about data format");
myprintf(prgid,"waiting for first packet...");
rv=poll(&pfd, 1, 1000); /* Timeout 1sec */
if(rv!=1) {
  sprintf(strlog,"TIMEOUT (rv= %d) : No packet coming from hardware : stopping everything !",rv);
  log_info(prgid,strlog);
  run=0; db->run=0; pthread_exit(NULL);
}
ptr = (unsigned char *)&(*packet);
//printf("net_thread : about to call recv(sock, ptr, MAX_PACKET_SIZE, 0)\n");
rv = recv(sock, ptr, MAX_PACKET_SIZE, 0);
// decode from first packet
packetsize = rv;
ts = (uint64_t)(*packet).header.TIMESTAMP;
bsn = (uint64_t)(*packet).header.BLOCK_SEQUENCE_NUMBER;
sprintf(strlog,"just got packet TS= %lu BSN= %lu",ts,bsn); log_info(prgid,strlog);
lastts = ts; lastbsn = bsn;
nofblocks = (int)(*packet).header.NOF_BLOCKS;
possiblediffbsn1=nofblocks;
possiblediffbsn2= nofblocks - (200000000/1024);
possiblediffbsn3= nofblocks - (200000000/1024 +1);
nobpb = (int)(*packet).header.NOF_BEAMLET_LO + ((*packet).header.NOF_BEAMLET_HI << 8);
bm = (int)(*packet).header.BM;
if(bm==0)db->bytespersample=8;
if(bm==1)db->bytespersample=4;
if(bm==2)db->bytespersample=2;  bytes_per_sample = db->bytespersample;
if(bm==2) {
  sprintf(strlog,"bm=2 which is the 4bits mode currently not supported...  Exiting !"); log_info(prgid,strlog); myprintf(prgid,strlog);
  run=0; db->run=0; pthread_exit(NULL);
}
//packetsize = headersize+nofblocks*nobpb*db->bytespersample;
// putting relevant info in shared memory
db->nob = nofblocks; db->nobpb = nobpb; db->bm= bm;
sprintf(strlog," got and analyzed first packet"); log_info(prgid,strlog); myprintf(prgid,strlog);
sprintf(strlog,"  headersize= %d  NbrOfBlock= %d  NbrOfBlockPerBank= %d bm= %d  -> packetsize= %d",headersize,nofblocks,nobpb,bm,packetsize);
log_info(prgid,strlog); myprintf(prgid,strlog);
sprintf(strlog," the 3 possible diffBSN are %ld %ld %ld",possiblediffbsn1,possiblediffbsn2,possiblediffbsn3); log_info(prgid,strlog);
//mode = packet[offset];
//printf("SOURCE_INFO= %02x %02x\n",packet[offset+1],packet[offset+2]);

// calculate the size of the overlap
if(0<db->fftovlp) {
  db->fftovlp= db->fftlen/2;
  overlap= db->fftovlp; offset_idx_samp= overlap;
  sizeoverlap= db->fftovlp * db->nobpb * db->bytespersample;
  sprintf(strlog,"asked for overlap :  fftovlp= %d  sizeovelap= %d",db->fftovlp,sizeoverlap); log_info(prgid,strlog);
}
// 'nofblocks' is forced (nber of time samples per packet)
// 'fftlen' is forced by the user
// determine 'nfft' max to fill 'nobpb' channels within DATABUF_SIZE
fftlenc=db->fftlen;
nfft2intc=db->nfft2int;
max_samp_per_buffer = DATABUF_SIZE*1024*1024 / ( nobpb * bytes_per_sample );
nfftc = max_samp_per_buffer/fftlenc;	// max nfft
//sprintf(strlog,"with NbrOfBlockPerBank(nbchan)= %d and bytes_per_sample= %d we derive max_samp_per_buffer= %d and nfftc= %d",nobpb,bytes_per_sample,max_samp_per_buffer,nfftc); log_info(prgid,strlog);
if(0<db->fftovlp) {
  nfftc--;
  sprintf(strlog,"asked for overlap :   max nber of FFT is reduced by 1 at %d",nfftc); log_info(prgid,strlog);
}
sprintf(strlog,"user wants  fftlen= %d nfft2int= %d  nofblocks= %d  determining nfft downwards from %d",fftlenc,nfft2intc,nofblocks,nfftc); log_info(prgid,strlog);
sprintf(strlog,"  fftlen= %d  nfft2int= %d  nofblocks= %d",fftlenc,nfft2intc,nofblocks); myprintf(prgid,strlog);
sprintf(strlog,"  max_samp_per_buffer= %d",max_samp_per_buffer); myprintf(prgid,strlog);
if(db->flgrawrt) {
  myprintf(prgid,"determining 'nfft' such as 'nfft*fftlen' is a multiple of nofblocks...");
  log_info(prgid,"determining 'nfft' such as 'nfft*fftlen' is a multiple of nofblocks...");
  while ( ((nfftc*fftlenc)%nofblocks != 0) ) nfftc--;
  sprintf(strlog,"  -> nfft= %d",nfftc); myprintf(prgid,strlog);
} else {
  sprintf(strlog," determining 'nfft' such as 'nfft*fftlen' is a multiple of nofblocks..."); log_info(prgid,strlog); myprintf(prgid,strlog);
  sprintf(strlog,"   and 'nfft' is a multiple of 'nfft2int'"); log_info(prgid,strlog); myprintf(prgid,strlog);
  while ( ((nfftc*fftlenc)%nofblocks != 0) || (nfftc%nfft2intc != 0) ) nfftc--;
  sprintf(strlog,"  -> nfft= %d",nfftc); myprintf(prgid,strlog);
}
db->nfft= nfftc;
samp_per_buffer = db->nfft*db->fftlen;
f=(float)(samp_per_buffer)/(float)max_samp_per_buffer;
sprintf(strlog," -> nfft= %d  used size of each buffer is %d bytes which is a fraction= %.4f",db->nfft,samp_per_buffer*nobpb*bytes_per_sample,f);
log_info(prgid,strlog);
sprintf(strlog,"samp_per_buffer= %d",samp_per_buffer); myprintf(prgid,strlog);
sprintf(strlog,"used size of each buffer =  %d bytes (frac= %.2f)",nobpb*db->nfft*db->fftlen*bytes_per_sample,f); myprintf(prgid,strlog);
if(nfftc==0) {
  sprintf(strlog,"%s : UNABLE to find a proper nfft. EXITING...",prgid); log_info(prgid,strlog); myprintf(prgid,strlog);
  log_info(prgid,"UNABLE to find a proper nfft. EXITING...");
  run=0; db->run=0; pthread_exit(NULL);
}

// -- first flush --
time2flush=5.0;
time2flush=2.0;
sprintf(strlog,"throwing away UDP packet for %.1lf sec for setup (fft_thread memory allocation and UDP buffer flush...)",time2flush);
log_info(prgid,strlog);
i=0; ib=0; dp=dateheurec(date,heure); df=dateheurec(date,heure);
while((df-dp)<time2flush) {
  rv = poll(&pfd, 1, 1000);
  rv = recv(sock, packet, MAX_PACKET_SIZE, 0);
  df=dateheurec(date,heure); i++;
  bsn = (uint64_t)(*packet).header.BLOCK_SEQUENCE_NUMBER; diffbsn = bsn-lastbsn; lastbsn = bsn;
  if ( (diffbsn==possiblediffbsn1) || (diffbsn==possiblediffbsn2) || (diffbsn==possiblediffbsn3) ) {
    // everything is fine !
    ts = (uint64_t)(*packet).header.TIMESTAMP;
    if(diffbsn==possiblediffbsn2) {
      sprintf(strlog," got round-SECOND crossing ts= %11lu bsn= %6d  diffbsn= %d",ts,bsn,diffbsn); log_info(prgid,strlog);
      if((ts-1)%2 ==0) startoneven= 0; else startoneven= 1;	// inverted on Apr 6, 2019 !
    }
    if(diffbsn==possiblediffbsn3) { sprintf(strlog," got round-SECOND crossing ts= %11lu bsn= %6d  diffbsn= %d",ts,bsn,diffbsn); log_info(prgid,strlog); }
  } else { ib++; }
}
gettimeofday(&tv,NULL); datetime=gmtime(&tv.tv_sec); currenttime2ts= (uint64_t)mktime(datetime);
lastts = (uint64_t)(*packet).header.TIMESTAMP; lastbsn = (uint64_t)(*packet).header.BLOCK_SEQUENCE_NUMBER;
sprintf(strlog,"Within %.1lf sec received and rejected %d packets (missing= %d MissFrac= %.3f)",time2flush,i,ib,(float)ib/(float)i); log_info(prgid,strlog);
sprintf(strlog,"  last received is TS= %lu  BSN= %lu  while current time as TS= %lu",lastts,lastbsn,currenttime2ts); log_info(prgid,strlog);
sprintf(strlog,"  startoneven= %d",startoneven); log_info(prgid,strlog);

// -- second FLUSH --
time2flush=1.0;
sprintf(strlog,"throwing away UDP packet for %.1lf sec for setup (fft_thread memory allocation and UDP buffer flush...)",time2flush);
log_info(prgid,strlog);
i=0; ib=0; dp=dateheurec(date,heure); df=dateheurec(date,heure);
while((df-dp)<time2flush) {
  rv = poll(&pfd, 1, 1000);
  rv = recv(sock, packet, MAX_PACKET_SIZE, 0);
  df=dateheurec(date,heure); i++;
  bsn = (uint64_t)(*packet).header.BLOCK_SEQUENCE_NUMBER; diffbsn = bsn-lastbsn; lastbsn = bsn;
  if ( (diffbsn==possiblediffbsn1) || (diffbsn==possiblediffbsn2) || (diffbsn==possiblediffbsn3) ) {
    // everything is fine !
  } else { ib++; }
}
gettimeofday(&tv,NULL); datetime=gmtime(&tv.tv_sec); currenttime2ts= (uint64_t)mktime(datetime);
lastts = (uint64_t)(*packet).header.TIMESTAMP; lastbsn = (uint64_t)(*packet).header.BLOCK_SEQUENCE_NUMBER;
sprintf(strlog,"Within %.1lf sec received and rejected %d packets (missing= %d MissFrac= %.3f)",time2flush,i,ib,(float)ib/(float)i); log_info(prgid,strlog);
sprintf(strlog,"  last received is TS= %lu  BSN= %lu  while current time as TS= %lu",lastts,lastbsn,currenttime2ts); log_info(prgid,strlog);
if(lastts!=currenttime2ts) { sprintf(strlog,"process node and hardware sending data apparenty NOT SYNCED !"); log_info(prgid,strlog); }

// information
sprintf(strlog,"Will listen to UDP packets on port %d",rcv_port); log_info(prgid,strlog);

if(0<db->startts) {
 // waiting for a specific TS packet (as well as a 1-pps crossing)
 sprintf(strlog,"Waiting for the specific TS= %d ... which is ~%lu sec",db->startts,db->startts-lastts); log_info(prgid,strlog); myprintf(prgid,strlog);
 ts=0;
 while( ts<db->startts ) {
   rv = poll(&pfd, 1, 1000);
   rv = recv(sock, packet, MAX_PACKET_SIZE, 0);
   ts = (uint64_t)(*packet).header.TIMESTAMP + 1;
 }
 sprintf(strlog,"Just got the previous TS = %lu",ts-1); log_info(prgid,strlog);
}
 // waiting for the next soft-1pps crossing
bsn=0;
while(bsn<=(195312-nofblocks)) {
  rv = poll(&pfd, 1, 1000);
  rv = recv(sock, packet, MAX_PACKET_SIZE, 0);
  bsn = (uint64_t)(*packet).header.BLOCK_SEQUENCE_NUMBER;
}
ts = (uint64_t)(*packet).header.TIMESTAMP;
log_info(prgid,"just got the last packet before a round-SECOND crossing");
sprintf(strlog,"  with TS= %lu  BSN= %lu",ts,bsn); log_info(prgid,strlog);
lastts= ts; lastbsn= bsn;
sprintf(strlog,"Starting acquisition and processing..."); myprintf(prgid,strlog);

sprintf(strlog,"Now filling block %d at %p",db->cur_buf,data); log_info(prgid,strlog);
effective_idx_samp = 0L;
current_effective_idx_samp = 0;
db->eisb[0]=0;	// not really usefull !
db->tsb[0]=0; db->bsnb[0]=0;
db->spectra_eis=0;
run=1; db->run=1;
while(db->run) {

  // poll (waiting for data ready with a timeout of 1 sec)
  rv = poll(&pfd, 1, 1000); /* Timeout 1sec */
  if (rv!=1) {
    if (rv==0) {
      printf("%s> poll() : TIMEOUT !\n",prgid); continue;
    } else {
      printf("%s> Error poll()\n",prgid); pthread_exit(NULL);
    }
  }

  rv = recv(sock, packet, MAX_PACKET_SIZE, 0);
  if (rv==-1) {
    printf("%s> Error recv -1 (curbuf= %d)\n",prgid,db->cur_buf);
    continue;
  } else {
    if (rv!=packetsize) {
      printf("%s> Error recv size= %d while packetsize= %d\n",prgid,rv,packetsize);
    } else {
      //printf("net_thread> recv OK\n");
    }
  }
  // Extract the sequence number
  ts = (uint64_t)(*packet).header.TIMESTAMP;
  bsn = (uint64_t)(*packet).header.BLOCK_SEQUENCE_NUMBER;
  nofblocksc = (int)(*packet).header.NOF_BLOCKS;
  if(nofblocksc!=nofblocks) { sprintf("BEWARE nofblocks changed from %d to %%d !!!",nofblocks,nofblocksc); log_info(prgid,strlog); }
  db->ts = ts; db->bsn = bsn;
  if(current_effective_idx_samp == 0) { db->tsb[db->cur_buf]=ts; db->bsnb[db->cur_buf]=bsn; }

  //printf("net_thread> one packet is received !\n");
  //printf("packet  = "); for(i=0;i<80;i++) printf(" %02x",packet[offset+i]); printf("\n");
  //printf("TIMESTAMP  = "); for(i=0;i<4;i++) printf(" %02x",packet[offset+8+i]); printf(" = %lu\n",ts);

  if(lastts!=ts){sprintf(strlog,"Got packet ts= %lu  bsn=%lu",ts,bsn); myprintf(prgid,strlog); }
  // Test if no packet lost
  diffbsn = bsn-lastbsn;  diffts = ts-lastts;
  if ( (diffbsn==possiblediffbsn1) || (diffbsn==possiblediffbsn2) || (diffbsn==possiblediffbsn3) ) {
    // EVERYTHING IS FINE !
    // raw copy, transpose is done in the GPU with CUBLAS... no longer with CUBLAS : it's now a specific kernel !
    iptr = (char *)&packet->beamlet;
    optr = data + (offset_idx_samp+current_effective_idx_samp)*nobpb*bytes_per_sample;	// each time sample takes 'nobpb' x4 8/16-bits
    memcpy(optr, iptr, nobpb*nofblocks*bytes_per_sample);
    // next idx_samp
    effective_idx_samp += (uint64_t)nofblocks;
    current_effective_idx_samp += nofblocks;
    // remains to check that buffer is full or not... see below
  } else {	// if ( (diffbsn==possiblediffbsn1) || (diffbsn==possiblediffbsn2) || (diffbsn==possiblediffbsn3) )
   // WE LOST PACKETs !
   if(effective_idx_samp!=0L) {	// to avoid thinking that first packet missed the previous one
      // a packet is probably missing...
      sprintf(strlog,"missing packet(s) between TS/BSN= %11lu %6lu and %11lu %6lu (diff= %lu %ld)",lastts,lastbsn,ts,bsn,diffts,diffbsn);
      log_info(prgid,strlog);
      // initialization to determine nber of dropped samples
      currentdroppedsamp= 0; tsc= lastts; bsnc= lastbsn;
      if(startoneven) nbbsn= 195313 - lastts%2; else nbbsn= 195312 + lastts%2;
      while ( (tsc!=ts) || (bsnc!=bsn) ) {
        bsnc += (uint64_t)nofblocks;
        tsc += bsnc/nbbsn;
        bsnc = bsnc%nbbsn;
        currentdroppedsamp+=nofblocks;
        if(bsnc/nbbsn) if(startoneven) nbbsn= 195313 - tsc%2; else nbbsn= 195312 + tsc%2;
        if(ts<tsc) { log_info(prgid," probably failed to caculate the number of missed time samples"); break; }
      }
      sprintf(strlog,"  currentdroppedsamp= %d  current_effective_idx_samp= %d (+currentdroppedsamp= %ld) to compare with samp_per_buffer= %d",
       currentdroppedsamp,current_effective_idx_samp,(uint64_t)current_effective_idx_samp+currentdroppedsamp,samp_per_buffer); log_info(prgid,strlog);
      effective_idx_samp += (uint64_t)currentdroppedsamp;
      current_effective_idx_samp += currentdroppedsamp;
      if( current_effective_idx_samp < samp_per_buffer) {
        // there is stil room in the buffer to put the packet
        iptr = (char *)&packet->beamlet;
        optr = data + (offset_idx_samp+current_effective_idx_samp)*nobpb*bytes_per_sample;	// each time sample takes 'nobpb' x4 8/16-bits
        memcpy(optr, iptr, nobpb*nofblocks*bytes_per_sample);
        // next idx_samp for next packet !
        effective_idx_samp += (uint64_t)nofblocks;
        current_effective_idx_samp += nofblocks;
        db->dropblk+=(currentdroppedsamp/nofblocks-1);
      } else {
        // need to change buffer to start over with a brand new one
	// -- no need to deal with the overlap, this is impossible due to lack of data !
        databuf_set_filled(db, db->cur_buf);
        db->cur_buf= (db->cur_buf + 1) % db->n_block;             // go to next buffer
        databuf_wait_free(db, db->cur_buf);                       // waiting for the next buffer to be freed...
        db->eisb[db->cur_buf]=effective_idx_samp;
        oldata= data;
        data = databuf_data(db,db->cur_buf);                      // pointer to data of next buffer
        current_effective_idx_samp= 0;
        iptr = (char *)&packet->beamlet;
        optr = data + (offset_idx_samp+current_effective_idx_samp)*nobpb*bytes_per_sample;
        memcpy(optr, iptr, nobpb*nofblocks*bytes_per_sample);
      }
   }
  }
  ib++; ibt++;
  // if current buffer is full
  if(current_effective_idx_samp>=samp_per_buffer) {
      //printf("net_thread : filling a new buffer...  current_effective_idx_samp= %d  (samp_per_buffer= %d)\n",current_effective_idx_samp,samp_per_buffer);
      old_cur_buf= db->cur_buf;					// to postpone the "databuf_set_filled()" call
      db->cur_buf= (db->cur_buf + 1) % db->n_block;		// go to next buffer
      databuf_wait_free(db, db->cur_buf);			// waiting for the next buffer to be freed...
      db->eisb[db->cur_buf]=effective_idx_samp;
      oldata= data;						// pointer to data of previous buffer
      data = databuf_data(db,db->cur_buf);			// pointer to data of next buffer
      current_effective_idx_samp-=samp_per_buffer;		// should set current_effective_idx_samp to 0
      //printf("net_thread : now current_effective_idx_samp = %d\n",current_effective_idx_samp);
      sprintf(strlog,"Now filling block %d at %p",db->cur_buf,data); log_info(prgid,strlog);
      if(0<sizeoverlap) {	// TO BE CHECKED !
         iptr = oldata + samp_per_buffer*nobpb*bytes_per_sample - sizeoverlap;	// point to [END - sizeoverlap] of previous buffer
         optr = data;								// put data at the ACTUAL beginning of the buffer (to deal with potential overlap)
         memcpy(optr, iptr, sizeoverlap);
         //effective_idx_samp-= (uint64_t)db->fftovlp;				// update the effective sample index (substracting overlap)
         sprintf(strlog,"overlap, copying the last %d bytes of previous buffer to beginning of new block",sizeoverlap); log_info(prgid,strlog);
      }
      databuf_set_filled(db, old_cur_buf);			// effective call to "databuf_set_filled()" to allow processing it
      //printf(" changing buffer done !\n");
      ib=0;
  }		// if(current_effective_idx_samp>=samp_per_buffer)
  // uncomment below if you want to know when a new packet arrived
  //printf("packet  ts= %lu  bsn= %lu  eis= %d %d\n",ts,bsn,effective_idx_samp,current_effective_idx_samp);
  lastts = ts; lastbsn = bsn;
  db->eis = effective_idx_samp;	// this 2 variables needs to be uint64_t to be able to handle observations longer than 3hrs

  // INJECT simulated data
  // -- this piece of code was removed...

  // -- output of first data received
#if 1
  if(effective_idx_samp==0) {
    iptr = (char *)&packet->beamlet;
    //for(ichan=0;ichan<nobpb;ichan++) {
    if(bytes_per_sample==4) {
     for(ichan=0;ichan<16;ichan++) {
      sprintf(strlog,"ichan= %03d sample= 00 datapacket= %4d %4d %4d %4d",ichan,
       (char)*(iptr+0),(char)*(iptr+1),(char)*(iptr+2),(char)*(iptr+3)); iptr+=4; log_info(prgid,strlog);
     }
     iptr = (char *)&packet->beamlet + nobpb*bytes_per_sample;
     for(ichan=0;ichan<16;ichan++) {
      sprintf(strlog,"ichan= %03d sample= 01 datapacket= %4d %4d %4d %4d",ichan,
       (char)*(iptr+0),(char)*(iptr+1),(char)*(iptr+2),(char)*(iptr+3)); iptr+=4; log_info(prgid,strlog);
     }
    }
    if(bytes_per_sample==8) {
     for(ichan=0;ichan<16;ichan++) {
      sprintf(strlog,"ichan= %03d sample= 00 datapacket= %6d %6d %6d %6d",ichan,
       (short)*(iptr+0),(short)*(iptr+2),(short)*(iptr+4),(short)*(iptr+6)); iptr+=8; log_info(prgid,strlog);
     }
     iptr = (char *)&packet->beamlet + nobpb*bytes_per_sample;
     for(ichan=0;ichan<16;ichan++) {
      sprintf(strlog,"ichan= %03d sample= 01 datapacket= %6d %6d %6d %6d",ichan,
       (short)*(iptr+0),(short)*(iptr+2),(short)*(iptr+4),(short)*(iptr+6)); iptr+=8; log_info(prgid,strlog);
     }
    }
  }
#endif

// -- Transfer data --
// -- SOURCE has 10-16 2-pol-cplx samples for each freq channel
// -- -- the order is f0s0 f1s0 f2s0 ... fNs0, f0s1 f1s1 ... fNs1, f2s2 ... , f0s15 ... fNs15, f0s16 ... fNs16 (with N=nofblocks)
// -- we want the DESTINATION data re-ordered as
// -- -- f0s0 f0s1 ... f0SM f1s0 f1s1 ... f1sM f2s0 ... fNs0 fNs1 fNs2 ... fNsM (with M=nfft*fftlen very large)

#if 0
  // catch a quick spectra
  if( bsn < db->nob){
    sprintf(strlog,"Catching a raw low resolution spectra for TS= %lu BSN= %lu eis= %d...",ts,bsn,effective_idx_samp); log_info(prgid,strlog);
    db->spectra_eis = effective_idx_samp;
    // -- -- the order is f0s0 f1s0 f2s0 ... fNs0, f0s1 f1s1 ... fNs1, f2s2 ... , f0s15 ... fNs15, f0s16 ... fNs16 (with N=nofblocks)
    for(ichan=0;ichan<nobpb;ichan++) {
      db->spectra_p0[ichan]=0.0; db->spectra_p1[ichan]=0.0;
      //iptr = (char *)&packet[offset+headersize] + ichan*bytes_per_sample;
      iptr = (char *)&packet->beamlet + ichan*bytes_per_sample;
      for(isamp=0;isamp<nofblocks;isamp++) {
        //xr= (float)(128+(char)*(iptr+0)); xi= (float)(128+(char)*(iptr+1));
        //yr= (float)(128+(char)*(iptr+2)); yi= (float)(128+(char)*(iptr+3));
        xr= (float)((char)*(iptr+0)); xi= (float)((char)*(iptr+1));
        yr= (float)((char)*(iptr+2)); yi= (float)((char)*(iptr+3));
        //if(ichan<3) printf("ichan= %d  xr xi yr yi= %f %f %f %f\n",ichan,xr,xi,yr,yi);
        db->spectra_p0[ichan] += xr*xr + xi*xi;  db->spectra_p1[ichan] += yr*yr + yi*yi;
        iptr += nobpb*bytes_per_sample;						// jump over the ~200 freq chan to go to next time sample
      }
    }
    nofblocksc = (int)(*packet).header.NOF_BLOCKS;
    nobpbc = (int)(*packet).header.NOF_BEAMLET_LO + ((*packet).header.NOF_BEAMLET_HI << 8);
    bmc = (int)(*packet).header.BM;
    if( (nofblocksc!=nofblocks) || (nobpbc!=nobpb) || (bmc!=bm) ) {
      sprintf(strlog,"Checked nofblocks, nobpb and mb ... they CHANGED !!! (%d, %d, %d)",nofblocksc,nobpbc,bmc); log_info(prgid,strlog);
    }
    //log_info("net_thread"," ... done");
  }
#endif

  // stop the observation if run was SET to ZERO !
  if(db->run==0) {
    sprintf(strlog,"Catched run=0... stopping !"); log_info(prgid,strlog);
  }

  // stop the observation if time is over !
  if((db->eis*db->rawtimeres) > db->tobs) {
    sprintf(strlog,"Duration time is over (duration= %lf, tobs= %lf)",(db->eis*db->rawtimeres),db->tobs); log_info(prgid,strlog);
    sprintf(strlog,"End of the observation..."); myprintf(prgid,strlog);
    mysleep(1.0); run=0; db->run=0;
  }
}

log_info(prgid,"Closing UDP socket");
close(sock);
sprintf(strlog,"Got %d packets, %d dropped (%.6f) on %s and UDPport %d",ibt,db->dropblk,(float)db->dropblk/(float)(ibt+db->dropblk),hostname,rcv_port);
log_info(prgid,strlog); myprintf(prgid,strlog);
log_info(prgid,"Exiting net_thread ..."); myprintf(prgid,"Exiting net_thread ...");
pthread_exit(NULL);
}
#if 0
  if(flgsimulated) {
    if(effective_idx_samp==0) {
      sprintf(strlog,"INJECT SIMULATED DATA"); log_info(prgid,strlog);
      for(isamp=0;isamp<nofblocks;isamp++) simulated[isamp]=(int)(96.0*sinf(6.2831853*(float)(isamp+1)/nofblocks));
      sprintf(strlog,"  simulated[]= ");
      for(isamp=0;isamp<nofblocks;isamp++) { sprintf(string," %4d",simulated[isamp]); strcat(strlog,string); }
      log_info(prgid,strlog);
    }
    k=0;
    ptr = (char *)&packet->beamlet;
    //printf("INJECT SIMULATED DATA ichan=%d %d\n",((effective_idx_samp/(fftlenc*nfft2intc))%16),effective_idx_samp);
    // -- RECALL : the order is f0s0 f1s0 f2s0 ... fNs0, f0s1 f1s1 ... fNs1, f2s2 ... , f0s15 ... fNs15, f0s16 ... fNs16 (with N=nofblocks)
    for(isamp=0;isamp<nofblocks;isamp++) {
     for(ichan=0;ichan<nobpb;ichan++) {
       if( (ichan == 4) || (ichan == 6) ) {
       //if( ichan == ((effective_idx_samp/(fftlenc*nfft2intc))%16) ) {
        //*ptr += simulated[isamp]; ptr++; *ptr += simulated[isamp]; ptr++;
        if (ichan == 4) {
        *ptr = 0; ptr++; *ptr = 0; ptr++;
        ptr++; ptr++;
        }
        if (ichan == 6) {
        ptr++; ptr++;
        *ptr = 0; ptr++; *ptr = 0; ptr++;
        }
        //ptr++; ptr++;
        //*ptr += simulated[isamp]; ptr++; *ptr += simulated[isamp]; ptr++;
       } else {
       // *ptr = 0; ptr++; *ptr = 0; ptr++;
       // *ptr = 0; ptr++; *ptr = 0; ptr++;
         ptr++; ptr++; ptr++; ptr++;
       }
     }
    }
    if(effective_idx_samp==0)log_info(prgid,"  simulated data INJECTED !");
  }
#endif
