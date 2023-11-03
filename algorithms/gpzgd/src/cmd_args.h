#ifndef CMD_ARGS_H
#define CMD_ARGS_H

#ifdef  __cplusplus
extern "C" {
#endif

    #include <stdint.h>

    #include "gp.h"

    struct main_parameters {
        char *   rng_seed_file;
        uint32_t rng_seed_offset;
    };

    struct main_parameters *default_main_parameters();
    void free_main_parameters(struct main_parameters *main_params);

    void load_config(int argc, char **argv, struct main_parameters *main_params, struct gp_parameters *params);

#ifdef  __cplusplus
}
#endif

#endif
