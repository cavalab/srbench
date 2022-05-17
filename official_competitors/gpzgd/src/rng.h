#ifndef RNG_H
#define RNG_H

#ifdef  __cplusplus
extern "C" {
#endif

    #include <stdio.h>
    #include <stdint.h>

    typedef uint64_t rng_state[4];

    void seed_rng(const char *const src, uint32_t offset, rng_state *state_ptr);

    void spawn_rng(rng_state parent, rng_state *state_ptr);

    double next_rnd(rng_state state);

    double next_rnd_gauss(double mu, double sd, rng_state state);

#ifdef  __cplusplus
}
#endif

#endif
