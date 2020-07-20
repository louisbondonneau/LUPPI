/*
 *  adapted from GUPPI
 *  Logging functions
 */

#ifndef _LOGGING_H
#define _LOGGING_H

// -- Some exit codes --
#define OK          0
#define TIMEOUT     1 // Call timed out 
#define ERR_GEN    -1 // Super non-informative
#define ERR_SYS    -2 // Failed system call
#define ERR_PARAM  -3 // Parameter out of range
#define ERR_KEY    -4 // Requested key doesn't exist
#define ERR_PACKET -5 // Unexpected packet size

#define LOG_FILENAME "/data/data2/luppi"


#ifdef __cplusplus
extern"C" {
#endif
// -- To log infos to LOG_FILENAME -- 
void log_info(const char *name, const char *msg);

// -- To log errors to LOG_FILENAME -- 
void log_error(const char *name, const char *msg);

// -- To log errors to LOG_FILENAME -- 
void log_warn(const char *name, const char *msg);
#ifdef __cplusplus
}
#endif

#endif
