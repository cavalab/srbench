#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>
#include <errno.h>

#include "alloc.h"
#include "cmd_args.h"
#include "gp.h"
#include "readline.h"
#include "rng.h"

static void load_data(char *src, double ***X_ptr, double **t_ptr, uint32_t *n_samples_ptr, uint16_t *n_feat_ptr)
{
    FILE *data = fopen(src, "r");

    if (data == NULL) {
        fprintf(stderr,
                "ERROR (%s:%d): Problem opening data file. Reason: %d (%s). Quitting.\n",
                __FILE__, __LINE__,
                errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    char *buffer = NULL;
    size_t bufsz = 0;
    char *line = next_line(&buffer, &bufsz, data);

    if (line == NULL) {
        fprintf(stderr,
                "ERROR (%s:%d): Problem reading data file header. Quitting.\n",
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    uint32_t n_samples;
    uint16_t n_feat;
    if (sscanf(line, "%u %hu", &n_samples, &n_feat) != 2) {
        fprintf(stderr,
                "ERROR (%s:%d): Problem reading data file header. Reason: %d (%s). Quitting.\n",
                __FILE__, __LINE__,
                errno, strerror(errno));
        exit(EXIT_FAILURE);
    }
    n_feat--; /* subtract one for the response! */

    double **X = MALLOC(n_feat, sizeof(double *));
    X[0] = MALLOC(n_feat * n_samples, sizeof(double));
    for (uint16_t j = 0; j < n_feat; ++j) X[j] = X[0] + j * n_samples;

    double *t = MALLOC(n_samples, sizeof(double));

    for (uint32_t i = 0; i < n_samples; ++i) {
        line = next_line(&buffer, &bufsz, data);
        char *tok = strtok(line, "\t ");
        for (uint16_t j = 0; j < n_feat; ++j) {
            X[j][i] = atof(tok);
            tok = strtok(NULL, "\t ");
        }
        t[i] = atof(tok);
    }

    *X_ptr = X;
    *t_ptr = t;
    *n_samples_ptr = n_samples;
    *n_feat_ptr = n_feat;

    free(buffer);
    fclose(data);
}

static double mse(double *t, double *y, uint32_t n_samples)
{
    double mse = 0;

    for (uint32_t i = 0; i < n_samples; ++i) {
        double rsqr = (t[i] - y[i]) * (t[i] - y[i]);
        mse += (rsqr - mse) / (i + 1);
    }

    return mse;
}

int main(int argc __attribute__((unused)), char **argv)
{
    struct main_parameters *main_params = default_main_parameters();
    struct gp_parameters *params = gp_default_parameters();

    double **X, *t, *y;
    uint32_t n_samples;
    uint16_t n_feat;

    /* first read in all the command line parameters and configuration files */
    load_config(argc - 2, argv + 2, main_params, params);

    seed_rng(main_params->rng_seed_file, main_params->rng_seed_offset, &(params->random_state));

    /* load problem */
    load_data(argv[1], &X, &t, &n_samples, &n_feat);

    struct gp *ind = gp_evolve(params, X, t, n_samples, n_feat);

    y = MALLOC(n_samples, sizeof(double));
    gp_predict(ind, X, n_samples, y);

    gp_print(stdout, ind);
    fprintf(stdout, ";%u;%f", gp_size(ind), mse(t, y, n_samples));

    free(y);
    gp_free(ind);

    free(t);
    free(X[0]);
    free(X);

    gp_free_parameters(params);
    free_main_parameters(main_params);

    return EXIT_SUCCESS;
}
