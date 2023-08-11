#include <stdlib.h>
#include <stdio.h>

#include <errno.h>
#include <libgen.h>
#include <math.h>
#include <string.h>

#include "alloc.h"
#include "cmd_args.h"
#include "gp.h"
#include "readline.h"

static void parse_parameters(char *params_file, struct main_parameters *main_params, struct gp_parameters *params);

static void parse_param(char *line, char **key, char **value)
{
    *key = line;

    while (*line != '=') line++;
    *line = '\0';

    *value = line + 1;

    *key = trim(*key);
    *value = trim(*value);
}

static void process_parameter(char *param, struct main_parameters *main_params, struct gp_parameters *params)
{
    char *line, *key, *value;

    line = MALLOC(strlen(param) + 1, sizeof(char));
    strcpy(line, param);

    parse_param(line, &key, &value);

    if (strncmp(key, "rng_seed_file", 13) == 0) {
        free(main_params->rng_seed_file);
        main_params->rng_seed_file = strdup(value);
    } else if (strncmp(key, "rng_seed_offset", 15) == 0) {
        main_params->rng_seed_offset = atoi(value);
    } else if (strncmp(key, "rng_seed", 8) == 0) {
        free(main_params->rng_seed_file);
        main_params->rng_seed_file = strdup("");
        main_params->rng_seed_offset = atoi(value);
    } else if (strncmp(key, "timeout", 7) == 0) {
        params->timeout = atoi(value);
    } else if (strncmp(key, "validation_prop", 15) == 0) {
        params->validation_prop = atof(value);
    } else if (strncmp(key, "pop_size", 8) == 0) {
        params->pop_size = atoi(value);
    } else if (strncmp(key, "generations", 11) == 0) {
        params->generations = atoi(value);
    } else if (strncmp(key, "elitism_rate", 12) == 0) {
        params->elitism_rate = atof(value);
    } else if (strncmp(key, "tournament_size", 15) == 0) {
        params->tournament_size = atof(value);
    } else if (strncmp(key, "crossover_rate", 14) == 0) {
        params->crossover_rate = atof(value);
    } else if (strncmp(key, "point_mutation_rate", 19) == 0) {
        params->point_mutation_rate = atof(value);
    } else if (strncmp(key, "sub_mutation_rate", 17) == 0) {
        params->sub_mutation_rate = atof(value);
    } else if (strncmp(key, "min_tree_init", 13) == 0) {
        params->min_tree_init = atoi(value);
    } else if (strncmp(key, "max_tree_init", 13) == 0) {
        params->max_tree_init = atoi(value);
    } else if (strncmp(key, "max_tree_depth", 14) == 0) {
        params->max_tree_depth = atoi(value);
    } else if (strncmp(key, "max_tree_nodes", 14) == 0) {
        params->max_tree_nodes = atoi(value);
    } else if (strncmp(key, "opset", 5) == 0) {
        gp_init_function_set(params, value);
    } else if (strncmp(key, "standardise", 11) == 0) {
        params->standardise = *value == 'Y' || *value == 'y';
    } else if (strncmp(key, "standardize", 11) == 0) {
        params->standardise = *value == 'Y' || *value == 'y';
    } else if (strncmp(key, "coef_op", 7) == 0) {
        params->coef_op = *value == 'Y' || *value == 'y';
    } else if (strncmp(key, "learning_rate", 13) == 0) {
        params->learning_rate = atof(value);
    } else if (strncmp(key, "learning_epochs", 15) == 0) {
        params->learning_epochs = atoi(value);
    } else if (strncmp(key, "mutation_sigma", 14) == 0) {
        params->mutation_sigma = atof(value);
    } else {
        fprintf(stderr, "WARNING: Unknown parameter: %s\n", key);
    }

    free(line);
}

static void parse_parameters(char *params_file, struct main_parameters *main_params, struct gp_parameters *params)
{
    FILE *input;
    char *buffer = NULL, *line = NULL, *base_file, *base_path, *base_config;
    size_t bufsz = 0;

    input = fopen(params_file, "r");
    if (input == NULL) {
        fprintf(stderr, "%s:%d - WARNING: Error loading config file %s. Reason: %d (%s)\n",
                __FILE__, __LINE__, params_file, errno, strerror(errno));
        return;
    }
    line = next_line(&buffer, &bufsz, input);
    while (!feof(input)) {
       /* if an include directive, then find path of current params file, and load relative to that */
        if (strlen(line) > 0) {
            if (strncmp(line, "include", 7)==0) {
                base_file = trim(line + 8);
                if (base_file[0] == '!') {
                    base_config = strdup(base_file + 1);
                } else {
                    base_path = dirname(strdup(params_file));
                    base_config = MALLOC(strlen(base_path) + strlen(base_file) + 2, sizeof(char));
                    sprintf(base_config, "%s/%s", base_path, base_file);
                    free(base_path);
                }
                parse_parameters(base_config, main_params, params);
                free(base_config);
            } else {
                process_parameter(line, main_params, params);
            }
        }
        line = next_line(&buffer, &bufsz, input);
    }
    fclose(input);
    free(buffer);
}

void load_config(int argc, char **argv, struct main_parameters *main_params, struct gp_parameters *params)
{
    int i = 0;

    while (i < argc) {
        if (strncmp(argv[i], "-p", 2) == 0) {
            process_parameter(argv[i + 1], main_params, params);
            i += 2;
        } else {
            parse_parameters(argv[i++], main_params, params);
        }
    }
}

struct main_parameters *default_main_parameters()
{
    struct main_parameters *p = MALLOC(1, sizeof(struct main_parameters));

    p->rng_seed_file   = NULL;
    p->rng_seed_offset = 0;

    return p;
}

void free_main_parameters(struct main_parameters *main_params)
{
    free(main_params->rng_seed_file);
    free(main_params);
}
