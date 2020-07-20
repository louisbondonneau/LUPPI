/* guppi_time.c
 *
 * Routines dealing with time conversion.
 */
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "slalib.h"
#include "logging.h"



int main(int argc, char *argv[])
// -------------------------------------------------------------------------------
{

  // -- Time parameters -- 
  int stt_imjd=0, stt_smjd=0;
  double stt_offs=0.0;
  get_current_mjd(&stt_imjd, &stt_smjd, &stt_offs);
  printf("imjd=%d smjd=%d offs=%lf\n", stt_imjd, stt_smjd, stt_offs);

}
