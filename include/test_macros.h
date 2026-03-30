#ifndef _TEST_MACROS_H
#define _TEST_MACROS_H

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define CHECK_ERR(err, TOL, OP)\
    ++test_ctr;\
    if(fabs(err) > TOL)\
    {\
        printf("%sTest[%2d/%2d] %s failed\n", KRED, test_ctr, total_tests, OP);\
        printf("%s",KNRM);\
        success_flag = false;\
    }\
    else\
    {\
        printf("%sTest[%2d/%2d] %s success\n", KGRN, test_ctr, total_tests, OP);\
        printf("%s",KNRM);\
    }

#define CHECK_LESS_THAN(a, b, OP)\
    ++test_ctr;\
    if(a >= b)\
    {\
        printf("%sTest[%2d/%2d] %s failed\n", KRED, test_ctr, total_tests, OP);\
        printf("%s",KNRM);\
        success_flag = false;\
    }\
    else\
    {\
        printf("%sTest[%2d/%2d] %s success\n",KGRN, test_ctr, total_tests, OP);\
        printf("%s",KNRM);\
    }

#define TESTS_START(ntests)\
    int total_tests = ntests;\
    printf("%sTESTS started\nThere are %d tests to pass%s\n",KCYN,total_tests,KNRM);\
    int test_ctr = 0;\
    bool success_flag = true;

#define TESTS_END\
    if(!success_flag)\
        printf("%sSorry, some tests failed%s\n",KRED, KNRM);\
    else\
        printf("%sCongrats !!!, You did it !\nAll tests passed%s\n",KCYN, KNRM);\

#endif
