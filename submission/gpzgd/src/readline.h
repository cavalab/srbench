#ifndef _READLINE_H
#define	_READLINE_H

#ifdef	__cplusplus
extern "C" {
#endif

    #include <stdlib.h>
    #include <stdio.h>

    char *next_line(char **buffer, size_t *sz, FILE *data);
    char *trim(char *str);

#ifdef	__cplusplus
}
#endif

#endif	/* _READLINE_H */
