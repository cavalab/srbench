#ifndef _ALLOC_H
#define _ALLOC_H

#ifdef  __cplusplus
extern "C" {
#endif

    #include <stdio.h>
    #include <stdlib.h>

    #include <stdint.h>
    #include <stdbool.h>
    #include <string.h>

    #define MALLOC(n, sz) reallocate(NULL, (n) * (sz), __FILE__, __LINE__, false)
    #define CALLOC(n, sz) reallocate(NULL, (n) * (sz), __FILE__, __LINE__, true)
    #define REALLOC(base, n, sz) reallocate((base), (n) * (sz), __FILE__, __LINE__, false)
    static void *reallocate(void *base, size_t sz, char *src_file, uint32_t line, bool clear)
    {
        base = realloc(base, sz);
        if (base == NULL) {
            fprintf(stderr, "%s:%u - ERROR: Failed to allocate memory\n",
                    src_file, line);
            exit(EXIT_FAILURE);
        }

        if (clear) memset(base, 0, sz);

        return base;
    }

#ifdef  __cplusplus
}
#endif

#endif
