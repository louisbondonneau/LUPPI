/* guppi_databuf.h
 *
 * Defines shared mem structure for data passing.
 * Includes routines to allocate / attach to shared
 * memory.
 */
#ifndef _DATABUF_H
#define _DATABUF_H

#include <sys/ipc.h>
#include <sys/sem.h>
#include <stdint.h>

#define DATABUF_KEY 12987498
#define DATABUF_NB 4 //8
#define DATABUF_SIZE 8196 //6144 // Size in MB

typedef struct {
    char data_type[64];            // Type of data in buffer 
    size_t struct_size;            // Size alloced for this struct (bytes) 
    int64_t block_size;             // Size of each data block (bytes) 
    size_t header_size;            // Size of each block header (bytes) 
    int shmid;                     // ID of this shared mem segment 
    int semid;                     // ID of locking semaphore set 
    int n_block;                   // Number of data blocks in buffer 
    int cur_buf;                   // Current buffer 
    int *cur_buf_offset;           // Current buffer offset 
} databuf;

typedef struct {
    databuf *db;                   // Pointer to overall shared mem databuf
    int block_idx;                 // Block index number in databuf
    size_t packet_data_size;       // Data size of each packet
    unsigned int packets_per_block;         // Total number of packets to go in the block
    int overlap_packets;           // Overlap between blocks in packets
    int npacket;                   // Number of packets filled so far
    int ndropped;                  // Number of dropped packets so far
    uint64_t packet_idx;           // Index of first packet number in block
    uint64_t last_pkt;             // Last packet seq number written to block
} datablock_stats;


/* union for semaphore ops.  Is this really needed? */
union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
    struct seminfo *__buf;
};

/* Create a new shared mem area with given params.  Returns 
 * pointer to the new area on success, or NULL on error.  Returns
 * error if an existing shmem area exists with the given shmid (or
 * if other errors occured trying to allocate it).
 */
databuf *databuf_create(int n_block, int64_t block_size, int gpu_id);

/* Return a pointer to a existing shmem segment with given id.
 * Returns error if segment does not exist 
 */
databuf *databuf_attach(int gpu_id);

/* Detach from shared mem segment */
int databuf_detach(databuf *d);

/* Clear out either the whole databuf (set all sems to 0, 
 * clear all header blocks) or a single FITS-style
 * header block.
 */
void databuf_clear(databuf *d);
void fitsbuf_clear(char *buf);

/* These return pointers to the header or data area for 
 * the given block_id.
 */
char *databuf_header(databuf *d, int block_id);
char *databuf_data(databuf *d, int block_id);
//char *databuf_data(databuf *d, int block_id);

/* Returns lock status for given block_id, or total for
 * whole array.
 */
int databuf_block_status(databuf *d, int block_id);
int databuf_total_status(databuf *d);

/* Databuf locking functions.  Each block in the buffer
 * can be marked as free or filled.  The "wait" functions
 * block until the specified state happens.  The "set" functions
 * put the buffer in the specified state, returning error if
 * it is already in that state.
 */
int databuf_wait_filled(databuf *d, int block_id);
int databuf_set_filled(databuf *d, int block_id);
int databuf_wait_free(databuf *d, int block_id);
int databuf_set_free(databuf *d, int block_id);


#endif
