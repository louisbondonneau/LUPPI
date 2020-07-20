/* polyco.c
 * routines to read/use polyco.dat
 */

#include "polyco.h"
#include "psrfits.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "logging.h"

int read_one_pc(FILE *f,  polyco *pc, const char *psr) {

    int i, j;
    char *rv;
    int ret;
    char buf[90], strlog[128];
    /* Read in polyco chunk */
    rv = fgets(buf, 90, f);
    if (rv==NULL) { return(-1); }
    strncpy(pc->psr, &buf[0], 10);  pc->psr[10] = '\0';

    pc->mjd = atoi(&buf[31]);
    pc->fmjd = atof(&buf[39]);
    pc->dm = atof(&buf[51]);
    pc->earthz4 = atof(&buf[73]);
    if ((rv=strchr(pc->psr, ' '))!=NULL) { *rv='\0'; }

    // -- check for correct psr - null psrname matches any --
    if (psr!=NULL) {
        if (strcmp(pc->psr, psr)!=0) {
	    sprintf(strlog, "Can not find '%s' in polyco file. Found src '%s'", psr, pc->psr);
	    log_error("read_one_pc", strlog); 
	    exit(1);
	}
    }

    rv = fgets(buf,90,f);
    if (rv==NULL) { return(-1); }
    pc->rphase_int = atoll(&buf[0]);
    pc->rphase = fmod(atof(&buf[0]),1.0);
    pc->f0 = atof(&buf[20]);
    pc->nsite = atoi(&buf[42]);
    pc->nmin = atoi(&buf[43]);
    pc->nc = atoi(&buf[50]);
    pc->rf = atof(&buf[55]);
    pc->used = 0;
    for (i=0; i<pc->nc/3 + (pc->nc%3)?1:0; i++) {
        rv=fgets(buf, 90, f);
        if (rv==NULL) { return(-1); }
        for (j=0; j<90; j++) { if (buf[j]=='D' || buf[j]=='d') buf[j]='e'; }
        ret=sscanf(buf, "%lf %lf %lf", 
                &(pc->c[3*i]), &(pc->c[3*i+1]), &(pc->c[3*i+2]));
        if (ret!=3) { return(-1); }
    }

    return(0);

}

int read_pc(FILE *f,  polyco *pc, const char *psr, int mjd, double fmjd) {

    /* Read through until we get to right psr, mjd */
    int done=0, nomatch=0;
    int i, j;
    char *rv;
    int ret;
    char buf[90];
    float tdiff;
    while (!done) {
        /* Read in polyco chunk */
        rv = fgets(buf, 90, f);
        if (rv==NULL) { done=1; nomatch=1; continue; }
        strncpy(pc->psr, &buf[0], 10);  pc->psr[10] = '\0';
        pc->mjd = atoi(&buf[31]);
        pc->fmjd = atof(&buf[39]);
        pc->dm = atof(&buf[51]);
        pc->earthz4 = atof(&buf[73]);
        if ((rv=strchr(pc->psr, ' '))!=NULL) { *rv='\0'; }
        rv = fgets(buf,90,f);
        pc->rphase = fmod(atof(&buf[0]),1.0);
        pc->f0 = atof(&buf[20]);
        pc->nsite = atoi(&buf[42]);
        pc->nmin = atoi(&buf[43]);
        pc->nc = atoi(&buf[50]);
        pc->rf = atof(&buf[55]);
        for (i=0; i<pc->nc/3 + (pc->nc%3)?1:0; i++) {
            rv=fgets(buf, 90, f);
            if (rv==NULL) { return(-1); }
            for (j=0; j<90; j++) { if (buf[j]=='D' || buf[j]=='d') buf[j]='e'; }
            ret=sscanf(buf, "%lf %lf %lf", 
                    &(pc->c[3*i]), &(pc->c[3*i+1]), &(pc->c[3*i+2]));
            if (ret!=3) { return(-1); }
        }
        /* check for correct psr - null psrname matches any */
        if (psr!=NULL) { if (strcmp(pc->psr, psr)!=0) { continue; } }
        tdiff = 1440.0*((double)(mjd-pc->mjd) + (fmjd-pc->fmjd));
        if (fabs(tdiff) > (float)pc->nmin/2.0) { continue; }
        done=1;
    }

    return(-1*nomatch);

}

/* Reads all polycos in a file, mallocs space for them, returns
 * number found
 */
int read_all_pc(FILE *f,  polyco **pc, const char *psr) {
    int rv, npc=0;
    do { 
        *pc = ( polyco *)realloc(*pc, sizeof( polyco) * (npc+1));
        rv = read_one_pc(f, &((*pc)[npc]), psr);
        npc++;
    } while (rv==0); 
    npc--; // Final "read" is really a error or EOF.
    return(npc);
}

/* Select appropriate polyco set */
int select_pc(const  polyco *pc, int npc, const char *psr,
        int imjd, double fmjd) {
    int ipc;
    const char *tmp = psr;
    if (psr!=NULL)
        if (tmp[0]=='J' || tmp[0]=='B') tmp++;
    // Verbose
    //fprintf(stderr, "Looking for polycos with src='%s' imjd=%d fmjd=%f\n",
    //        tmp, imjd, fmjd);
    for (ipc=0; ipc<npc; ipc++) {
        //fprintf(stderr, "  read src='%s' imjd=%d fmjd=%f span=%d\n",
        //        pc[ipc].psr, pc[ipc].mjd, pc[ipc].fmjd, pc[ipc].nmin);
        if (psr!=NULL) { if (strcmp(pc[ipc].psr,tmp)!=0) { continue; } }
        if (pc_out_of_range(&pc[ipc],imjd,fmjd)==0) { break; }
    }
    if (ipc<npc) { return(ipc); }
    return(-1);
}

/* Compute pulsar phase given polyco  and mjd */
double psr_phase(const  polyco *pc, int mjd, double fmjd, double *freq,
        long long *pulsenum) {
    double dt = 1440.0*((double)(mjd - pc->mjd)+(fmjd - pc->fmjd));
    int i;
    char strlog[256];
    //sprintf(strlog, "IN PSR PHASE  %.32lf",(double)pc->nmin/2.0-fabs(dt));
    //log_info("polyco", strlog);
    double phase = pc->c[pc->nc-1];
    double f = 0.0;
    
    //if (fabs(dt)>(double)pc->nmin/2.0) { 
    //    sprintf(strlog, "psr_phase return -1 %lf" ,fabs(dt)-(double)pc->nmin/2.0);
    //    log_error("polyco", strlog);
    //    return(-1.0);
    //    }
    for (i=pc->nc-1; i>0; i--) {
        phase = dt*(phase) + pc->c[i-1];
        f = dt*(f) + (double)i*pc->c[i];
    }
    f = pc->f0 + (1.0/60.0)*f;
    phase += pc->rphase + dt*60.0*pc->f0;
    if (freq!=NULL) { *freq = f; }
    if (pulsenum!=NULL) { 
        long long n = pc->rphase_int;
        n += (long long)(phase - fmod(phase,1.0));
        phase = fmod(phase,1.0);
        if (phase<0.0) { phase += 1.0; n--; }
        *pulsenum = n; 
    }
    //sprintf(strlog, "phase=%.16f   pc->c[pc->nc-1] = %.16f    dt = %.16f", phase, pc->c[pc->nc-1], dt*60);
    //log_info("polyco", strlog);
    return(phase);
}

double psr_fdot(const  polyco *pc, int mjd, double fmjd, double *fdot) {
    double dt = 1440.0*((double)(mjd-pc->mjd)+(fmjd-pc->fmjd));
    if (fabs(dt)>(double)pc->nmin/2.0) { return(-1.0); }
    double fd=0.0;
    int i;
    for (i=pc->nc-1; i>1; i--) {
        fd = dt*fd + ((double)i)*((double)i-1.0)*pc->c[i];
    }
    fd /= 60.0;
    if (fdot!=NULL) { *fdot=fd; }
    return(fd);
}

double psr_phase_avg(const  polyco *pc, int mjd, 
        double fmjd1, double fmjd2) {
    double dt1 = 1440.0*((double)(mjd-pc->mjd)+(fmjd1-pc->fmjd));
    double dt2 = 1440.0*((double)(mjd-pc->mjd)+(fmjd2-pc->fmjd));
    if (fabs(dt1)>(double)pc->nmin/2.0) { return(-1.0); }
    if (fabs(dt2)>(double)pc->nmin/2.0) { return(-1.0); }
    double pavg;
    int i;
    double tmp1=0.0, tmp2=0.0;
    for (i=pc->nc-1; i>=0; i--) {
        tmp1 = dt1*tmp1 + pc->c[i]/((double)i+1.0);
        tmp2 = dt2*tmp2 + pc->c[i]/((double)i+1.0);
    }
    tmp1 *= dt1; tmp2 *= dt2;
    pavg = (tmp2-tmp1)/(dt2-dt1) + pc->rphase + (dt1+dt2)*60.0*pc->f0/2.0;
    return(pavg);
}

int pc_range_check(const  polyco *pc, int mjd, double fmjd) {
    double dt;
    dt = (double)(mjd - pc->mjd) + (fmjd - pc->fmjd);
    dt *= 1440.0;
    if (dt < -1.0*(double)pc->nmin/2.0) { return(-1); }
    else if (dt > (double)pc->nmin/2.0) { return(1); }
    else { return(0); }
}

int pc_out_of_range(const  polyco *pc, int mjd, double fmjd) {
    double dt;
    dt = (double)(mjd - pc->mjd) + (fmjd - pc->fmjd);
    dt *= 1440.0;
    if (fabs(dt)>(double)pc->nmin/2.0) { return(1); }
    return(0);
}

int pc_out_of_range_sloppy(const  polyco *pc, int mjd, double fmjd, 
        double slop) {
    double dt;
    dt = (double)(mjd - pc->mjd) + (fmjd - pc->fmjd);
    dt *= 1440.0;
    if (fabs(dt)>slop*(double)pc->nmin/2.0) { return(1); }
    return(0);
}

/* Check whether or not two polyco s are the same */
int polycos_differ(const  polyco *p1, const  polyco *p2) {
    // Could add more tests as needed
    if (strncmp(p1->psr, p2->psr,15)!=0) return(1);
    if (p1->mjd!=p2->mjd) return(1);
    if (p1->fmjd!=p2->fmjd) return(1);
    if (p1->rf!=p2->rf) return(1);
    if (p1->nsite!=p2->nsite) return(1);
    if (p1->nmin!=p2->nmin) return(1);
    if (p1->nc!=p2->nc) return(1);
    return(0);
}

/* Convert telescope name to tempo code */
char telescope_name_to_code(const char *name) {

    /* Assume a 1-char input is already a code */
    if (strlen(name)==1) { return(name[0]); }

    /* Add to these as needed .. */
    if (strcasecmp(name, "GBT")==0) return('1');

    if (strcasecmp(name, "GB43m")==0) return('a');
    if (strcasecmp(name, "GB 43m")==0) return('a');
    if (strncasecmp(name, "GB140",5)==0) return('a');
    if (strncasecmp(name, "GB 140",6)==0) return('a');

    if (strcasecmp(name, "Arecibo")==0) return('3');
    if (strcasecmp(name, "AO")==0) return('3');

    if (strcasecmp(name, "NC")==0) return('f');
    if (strcasecmp(name, "Nancay")==0) return('f');
    if (strcasecmp(name, "NF")==0) return('n');
    if (strcasecmp(name, "NenuFAR")==0) return('n');
    if (strcasecmp(name, "nenufar")==0) return('n');
    if (strcasecmp(name, "NRT")==0) return('f');

    if (strcasecmp(name, "ATA")==0) return('j');

    /* Not found, return null */
    return('\0');
}

/* Generate polycos from a parfile */
/*
#define make_polycos_cleanup() do {\
    rv = chdir(origdir);\
    if(rv==-1) fprintf(stderr, "make_polycos: error system");\
    free(origdir);\
    rmdir(tmpdir);\
} while (0)
*/

 
void make_polycos_cleanup(char *polycofile) {
    //unlink("pulsar.par");
    unlink(polycofile);
} 

int make_polycos(const char *parfile, struct hdrinfo *hdr, char *src,  polyco **pc) {
  
    log_info("polyco", "make polyco");
    int rv;
    char strlog[128];
        
    /* Open parfile */
    FILE *pf = fopen(parfile, "r");
    if (pf==NULL) {
        sprintf(strlog,"make_polycos: Error opening parfile %s", parfile);
        log_error("polyco", strlog);
        return(-1);
    }
    char tmpdir[128];
    static mode_t mode = 0777;
    sprintf(tmpdir, POLYCODIR);
    rv = mkdir(tmpdir, mode);
    sprintf(tmpdir, "%s/BEAM%d", tmpdir, (int) atoi(hdr->beam));
    /* change to temp dir */
    //char *origdir = getcwd(NULL,0);
    rv = chdir(tmpdir);
    if(rv==-1) {
        sprintf(strlog, "Error system chdir");
        log_error("polyco", strlog);
    }

    /* Open temp dir */
    char fname[256];


    sprintf(fname, "%s/pulsar-%d-BEAM%d-%d.par", tmpdir, (int) hdr->scan_number, (int) atoi(hdr->beam), (int) hdr->fctr);
    FILE *fout = fopen(fname, "w");
    if (fout==NULL) {
        sprintf(strlog, "Error writing to temp dir %s",fname);
        log_error("polyco", strlog);
        fclose(pf);
        return(-1);
    }

    /* Get source name, copy file */
    char line[256], parsrc[32]="", *key, *val, *saveptr, *ptr;
    while (fgets(line,256,pf)!=NULL) {
        fprintf(fout, "%s", line);
        while ((ptr=strchr(line,'\t'))!=NULL) *ptr=' ';
        if ((ptr=strrchr(line,'\n')) != NULL) *ptr='\0'; 
        key = strtok_r(line, " ", &saveptr);
        val = strtok_r(NULL, " ", &saveptr);
        if (key==NULL || val==NULL) continue; 
        if (strncmp(key, "PSR", 3)==0) { 
            // J or B is bad here?
            if (val[0]=='J' || val[0]=='B') val++;
            strcpy(parsrc, val); 
        }
    }
    fclose(pf);
    fclose(fout);
    if (parsrc[0]=='\0') {
        sprintf(strlog, "Couldn't find source name in %s", parfile);
        log_error("polyco", strlog);
        return(-1);
    }
    if (src!=NULL) { strcpy(src,parsrc); }

    /* Get telescope character */
    char tcode = telescope_name_to_code(hdr->telescope);
    if (tcode=='\0') {
        sprintf(strlog, "Unrecognized telescope name (%s)", hdr->telescope);
        log_error("polyco", strlog);
        //make_polycos_cleanup();
        return(-1);
    }

#if 0
    /* Write tz.in */
    sprintf(fname, "tz.in-%d", (int) hdr->fctr);
    //sprintf(fname, "%s/tz.in-%d", tmpdir, (int) hdr->fctr);
    fout = fopen(fname, "w");
    if (fout==NULL) { 
        sprintf(strlog, "make_polycos: Error opening tz.in for write");
        log_error("polyco", strlog);
        //make_polycos_cleanup();
        return(-1);
    }
    //fprintf(fout, "%c 12 60 15 1400.0\n\n\n%s\n", tcode, parsrc);
    fprintf(fout, "%c 12 60 15 %.5f\n\n\n%s\n", tcode, hdr->fctr, parsrc);
    fclose(fout);
#endif
    // -- Set polyco filename --
    char polycofile[128];
    char OLDpolyco[128];
    sprintf(OLDpolyco, "%s/polyco_new.dat",tmpdir);
    sprintf(polycofile, "%s/polyco-%d-BEAM%d-%d.dat", tmpdir, (int) hdr->scan_number, (int) atoi(hdr->beam) , (int) hdr->fctr); //ADD LOUIS scan_numb and gpu_id

    /* Call tempo */
    int mjd0, mjd1;
    double mjd0b;
    mjd0 = (int)hdr->MJD_epoch;
    mjd0b = (double)(int)(hdr->MJD_epoch*1.e3);
    mjd0b = (mjd0b - (double)((int)mjd0b % 125))/1.e3;

    mjd1 = (int)(hdr->MJD_epoch + hdr->scanlen/86400.0 + 0.5);

    if (mjd1==mjd0) mjd1++;
    sprintf(line, "tempo -z -f %s -Z PSR=%s -Z SITE=%c -Z OUT=%s -Z START=%f -Z FREQ=%.1lf -Z TOBSH=12 -Z SPAN=15M", fname, parsrc, tcode,  polycofile, mjd0b , hdr->fctr);
    //sprintf(line, "tempo2 -f %s -polyco \" %f %f 60 15 8 %c %.6lf\" -tempo1", fname, mjd0-0.5, mjd0+1.5, tcode,  polycofile, hdr->fctr);

    log_info("polyco cmd line", line);
    printf("polyco: %s\n",line);
    rv = system(line);
    if(rv==-1) {
        sprintf(strlog, "Error system(line)");
        log_error("polyco", strlog);
    }	


    //int returnCode = rename(OLDpolyco ,polycofile);


    //if(returnCode == 0) {
    //   printf("polyco renamed successfully\n");
    //} else {
    //   printf("Error: unable to rename the polyco  %s to %s \n",OLDpolyco ,polycofile);
    //}



    /* Read polyco file */
    FILE *pcfile = fopen(polycofile, "r");
    if (pcfile==NULL) {
        sprintf(strlog, "Error reading polyco %s sleep for 3 sec and Im going to try again", polycofile);
        log_error("polyco", strlog);
        sleep(3);
        rv = system(line);
        if(rv==-1) {
            sprintf(strlog, "Error system(line) for the second time");
            log_error("polyco", strlog);
        }
        if (pcfile==NULL) {
            sprintf(strlog, "Error reading polyco for the second time %s", polycofile);
            log_error("polyco", strlog);
            return(-1);
        }
    }
    int npc = read_all_pc(pcfile, pc, parsrc);
    fclose(pcfile);

    sprintf(strlog, "Read %d polyco sets", npc);
    log_info("polyco", strlog);
    /* Clean up */


    if (npc>0) make_polycos_cleanup(polycofile);

    return(npc);
}

int make_const_polyco(double freq, const struct hdrinfo *hdr, polyco **pc) {
    *pc = realloc(*pc, sizeof(polyco));
    sprintf((*pc)[0].psr, "CONST");
    (*pc)[0].mjd = hdr->start_day;
    (*pc)[0].fmjd = hdr->start_sec/86400.0;
    (*pc)[0].rphase = 0.0;
    (*pc)[0].f0 = freq;
    (*pc)[0].nsite = 0;
    (*pc)[0].nmin = 20 * 24.0*60.0;
    (*pc)[0].nc = 1;
    (*pc)[0].rf = hdr->fctr;
    (*pc)[0].c[0] = 0.0;
    (*pc)[0].used = 0;
    return(1);
}
