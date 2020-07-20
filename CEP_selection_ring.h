#ifndef _CEP_H_
#define _CEP_H_


#include<stdio.h>
#include<math.h>

#define BM_c           (  1)
#define NOF_BEAMLET_c  (122)
#define NOF_BLOCKS_c   (  16)

#define NOF_BEAMLETBUF_c  (32)
#define NOF_BLOCKSBUF_c   (  32768)
#define NOF_STRUCTINBUFF_c   (  3)


#if   (BM_c == 0)
  typedef struct complex_t
  {
    int16_t re;
    int16_t im;
  } complex_t;
#elif (BM_c == 1)
  typedef struct complex_t
  {
    int8_t re;
    int8_t im;
  } complex_t;
#elif (BM_c == 2)
  typedef struct complex_t
  {
    int8_t re : 4,
           im : 4;
  } complex_t;

#endif


typedef struct beamlet_t
{
  complex_t X;
  complex_t Y;
} beamlet_t;

//typedef struct SOURCE_INFO_t
//{
//  uint16_t          RSP_ID: 5,
//                          : 1,
//                    ERR   : 1,
//                    F200M : 1,
//                    BM    : 2,
//                          : 2,
//           NOF_BEAMLET_HI : 4;
//} SOURCE_INFO_t;

typedef struct header_t
{
  uint64_t      VERSION_ID      :  8,
//                SOURCE_INFO     : 16,
                         RSP_ID :  5,
                                :  1,
                         ERR    :  1,
                         F200M  :  1,
                         BM     :  2,
                                :  2,
                NOF_BEAMLET_HI  :  4,
//                SOURCE_INFO     : 16,
                CONFIGURATION_ID:  8,
                NOF_BLOCKS_RESIZE      : 16,
                NOF_BEAMLET_LO  :  8,
                NOF_BLOCKS      :  8;
  uint32_t      TIMESTAMP;
  uint32_t      BLOCK_SEQUENCE_NUMBER;
} header_t;


typedef struct CEP_frame_t CEP_frame_t;
struct CEP_frame_t
{
  header_t  header;
  beamlet_t beamlet[NOF_BLOCKS_c][NOF_BEAMLET_c];
};




typedef struct BUF_frame_t BUF_frame_t;
struct BUF_frame_t
{
  header_t  header;
  beamlet_t beamlet[NOF_BLOCKSBUF_c][NOF_BEAMLETBUF_c];
};

typedef struct STATINBUFF_t
{
  int STAT;
} STATINBUFF_t;
  
  
typedef struct RingBuffer RingBuffer;
struct RingBuffer
{
  STATINBUFF_t STATINBUFF[NOF_STRUCTINBUFF_c];
  BUF_frame_t BUF_frame[NOF_STRUCTINBUFF_c];
};


#endif /* _CEP_H_ */
