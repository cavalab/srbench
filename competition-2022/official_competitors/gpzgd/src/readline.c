#include <stdio.h>
#include <stdlib.h>

#include <ctype.h>  /* needed for isspace */
#include <string.h> /* needed for string manipulation (e.g., strlen) */

static int readline(char **lineptr, size_t *n, FILE *stream)
{
    return getdelim(lineptr, n, '\n', stream);
}

char *trim(char *str)
{
    char *end;

    if (strlen(str) > 0) {
        while (isspace(*str)) str++;

        end = str + strlen(str) - 1;
        while (end > str && isspace(*end)) end--;

        *(end + 1) = '\0';
    }

    return str;
}

char *next_line(char **buffer, size_t *sz, FILE *data)
{
    char *line;
    readline(buffer, sz, data);
    line = *buffer;

    if (line == NULL) return NULL;

    if (strlen(line) > 0) {
        line[strcspn(*buffer, "\n")] = '\0'; /* chomp newline character, if present */
        line[strcspn(*buffer, "#")] = '\0';  /* chomp comments at end of line */
    }

    return trim(line);
}
