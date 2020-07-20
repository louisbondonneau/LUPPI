/* guppi_udp.h
 *
 * Functions dealing with setting up and 
 * receiving data through a UDP connection.
 */
#ifndef _UDP_H
#define _UDP_H

#include <sys/types.h>
#include <netdb.h>
#include <poll.h>

#include "psrfits.h"

#define MAX_PACKET_SIZE 9000

/* Struct to hold connection parameters */
typedef struct {

    /* Info needed from outside: */
    char sender[80];  /* Sender hostname */
    char dst_ip[80];  /* Sender hostname */
    char rcv_ip[80];  /* Sender hostname */
    int port;         /* Receive port */
    size_t packet_size;     /* Expected packet size, 0 = don't care */
    char packet_format[32]; /* Packet format */

    /* Derived from above: */
    int sock;                       /* Receive socket */
    struct addrinfo sender_addr;    /* Sender hostname/IP params */
    struct pollfd pfd;              /* Use to poll (wait) for avail data */
    struct sockaddr_in host;
} udp_params;

typedef struct {
    size_t packet_size;  /* packet size, bytes */
    uint64_t fpga_counter;
    uint64_t chan_id;
    uint64_t *last_fpga_counter;
    char data[MAX_PACKET_SIZE]; /* packet data */
} udp_packet;

int packet_init(udp_packet *b, struct psrfits *pf);

uint64_t udp_packet_seq_num(const udp_packet *packet);
uint16_t udp_packet_chan(const udp_packet *packet);
uint64_t udp_packet_IP_id(const udp_packet *packet);
int valide_packet(const udp_packet *packet);
char *udp_packet_data(const udp_packet *p);
size_t udp_packet_datasize(size_t packet_size);


//void udp_packet_data_copy(char *out, char *in);
/*
size_t parkes_udp_packet_datasize(size_t packet_size);
unsigned long long guppi_udp_packet_flags(const struct guppi_udp_packet *p);
*/
struct in_addr *udp_atoaddr(char *address);
/* Use sender and port fields in param struct to init
 * the other values, bind socket, etc.
 */
int udp_init(udp_params *params);
int udp_forward_init(udp_params *params);

/* Wait for available data on the UDP socket */
int udp_wait(udp_params *params); 

/* Read a packet */
int udp_recv(udp_params *params, udp_packet *packet);

/* Forward a packet to another IP*/
int udp_forward(udp_params *params, udp_packet *packet);

/* Read the polarization of the packet */
int udp_packet_polar(const udp_packet *packet);

/* Copy a guppi packet to the specified location in memory, 
 * expanding out missing channels for 1SFA packets.
 */
void udp_packet_data_copy(char *out, const udp_packet *p);

/* Copy and corner turn for baseband multichannel modes */
void udp_packet_data_copy_transpose(char *databuf, int nchan, unsigned block_pkt_idx, unsigned packets_per_block, const udp_packet *p);

/* Close out socket, etc */
int udp_close(udp_params *p);
#endif
