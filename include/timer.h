#ifndef _TIMER_H
#define _TIMER_H

#include <sys/time.h>
#include <vector>
#include <map>
#include <cstddef>
#include <string>

#define TIMING

#ifdef TIMING

extern std::vector<double> fn_timer;
extern std::vector<unsigned int> fn_counts;
extern std::vector<std::string> fn_name;
extern std::map<std::string, int> fn_map;

#define START_TIMER(region)\
    timeval start##region, end##region;\
    double start_tym##region, end_tym##region;\
    static int timer_id = -1;\
    if(timer_id < 0)\
    {\
        timer_id = fn_timer.size();\
        fn_timer.push_back(0);\
        fn_counts.push_back(0);\
        char *name##region;\
        asprintf(&name##region, "%s", #region);\
        fn_name.push_back(name##region);\
        fn_map[name##region] = timer_id;\
    }\
    gettimeofday(&start##region, NULL); \
    start_tym##region = start##region.tv_sec + start##region.tv_usec*1e-6;\

#define STOP_TIMER(region)\
    gettimeofday(&end##region, NULL);\
    end_tym##region = end##region.tv_sec + end##region.tv_usec*1e-6;\
    fn_timer[timer_id] += (end_tym##region-start_tym##region);\
    fn_counts[timer_id] += 1;


#define PRINT_TIME_SUMMARY\
    const char* line = "----------------------------------------------------------------------------------\n";\
    printf("\n%s \t\t\t\t  Timing Summary\n\n%s", line, line);\
    printf("%15s |\t %10s |\t %15s |\t %15s |\n%s", "FUNCTION", "COUNTS", "TOTAL TIME (s)", "TIME/COUNT (s)", line);\
    for(unsigned int i=0; i<fn_timer.size(); ++i)\
    {\
        printf("%15s |\t %10u |\t %.9e |\t %.9e |\n", fn_name[i].c_str(), fn_counts[i], fn_timer[i], fn_timer[i]/static_cast<double>(fn_counts[i]));\
    }\
    printf("%s\n",line);

#define GET_TIMER(time, region)\
{\
    char *name##region;\
    asprintf(&name##region, "%s", #region);\
    time = fn_timer[fn_map[name##region]];\
}

#else

#define START_TIMER(region)\

#define STOP_TIMER(region)\

#define GET_TIMER(time, region)\

#define PRINT_TIME_SUMMARY\


#endif

#endif
