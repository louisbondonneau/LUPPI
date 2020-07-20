#include <stdio.h>
#include <time.h>
#include "logging.h"


void log_info(const char *name, const char *msg) {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    char strlog[80];
    FILE *pfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );

    strftime (buffer,80,"%Y-%m-%d %H:%M:%S,000",timeinfo);
    sprintf(strlog,"%s_%d.log", LOG_FILENAME, (int)getpid());

    if((pfo=fopen(strlog,"a")) != NULL) {
        fprintf(pfo,"%s - %s - INFO : %s\n", buffer, name, msg);
        fclose(pfo);
    } else {
        fprintf(stderr,"%s - log_info - ERROR : Cannot open %s",buffer,strlog);
        fprintf(stderr,"%s - %s - INFO : %s", buffer, name, msg);
    }	
}


void log_error(const char *name, const char *msg) {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    char strlog[80];
    FILE *pfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );

    strftime (buffer,80,"%Y-%m-%d %H:%M:%S,000",timeinfo);
    sprintf(strlog,"%s_%d.log", LOG_FILENAME, (int)getpid());

    if((pfo=fopen(strlog,"a")) != NULL) {
        fprintf(pfo,"%s - %s - ERROR : %s\n", buffer, name, msg);
        fclose(pfo);
    } else {
        fprintf(stderr,"%s - log_error - ERROR : Cannot open %s",buffer,strlog);
        fprintf(stderr,"%s - %s - ERROR : %s", buffer, name, msg);
    }	
}

void log_warn(const char *name, const char *msg) {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    char strlog[80];
    FILE *pfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );

    strftime (buffer,80,"%Y-%m-%d %H:%M:%S,000",timeinfo);
    sprintf(strlog,"%s_%d.log", LOG_FILENAME, (int)getpid());

    if((pfo=fopen(strlog,"a")) != NULL) {
        fprintf(pfo,"%s - %s - WARNING : %s\n", buffer, name, msg);
        fclose(pfo);
    } else {
        fprintf(stderr,"%s - log_error - ERROR : Cannot open %s",buffer,strlog);
        fprintf(stderr,"%s - %s - WARNING : %s", buffer, name, msg);
    }	
}

