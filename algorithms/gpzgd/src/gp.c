/* #define _GNU_SOURCE */
#include <stdio.h>
#include <stdbool.h>

#include <string.h>
#include <math.h>
#include <float.h>

#include <unistd.h>
#include <signal.h>

#include "alloc.h"
#include "rng.h"

#include "gp.h"

#define OPERATOR(op) ((op) <= PINV)
#define BINARY_OPERATOR(op) (OPERATOR(op) && (op) <= PDIV)
#define UNARY_OPERATOR(op)  (OPERATOR(op) && (op)  > PDIV && (op) <= PINV)

#define BIG_NUMBER  1.0e+15
#define SMALL_LIMIT 1.0e-11
#define EXP_LIMIT   30
#define sane(d) (isfinite(d) && fabs(d) < BIG_NUMBER)

struct parse_tree;

typedef double   (*nodeop)(struct parse_tree *, double *);
typedef uint16_t (*nodead)(struct parse_tree *, double, double *, uint16_t);
typedef void     (*nodepr)(struct parse_tree *, FILE *);

static volatile bool interrupted = false;

struct parse_tree {
    nodeop op;
    nodead ad;
    nodepr pr;

    uint8_t  depth;
    uint16_t size;
    struct parse_tree *parent;

    struct parse_tree *arg0;
    struct parse_tree *arg1;

    int32_t i;
    double f;
};



struct gp {
    struct parse_tree *tree;

    uint16_t p; /* number of predictors */
    double *data;
    double *mu;
    double *s;
};

struct gp_parameters *gp_default_parameters()
{
    struct gp_parameters *params = MALLOC(1, sizeof(struct gp_parameters));

    memset(params->random_state, 0, sizeof(rng_state));

    params->validation_prop = 0.0;

    params->pop_size = 200;
    params->generations = 250;
    params->elitism_rate = 1;
    params->tournament_size = 3;
    params->crossover_rate = 0.3;
    params->sub_mutation_rate = 0.4;
    params->point_mutation_rate = 0.3;
    params->min_tree_init = 2;
    params->max_tree_init = 4;
    params->max_tree_depth = -1;
    params->max_tree_nodes = 50;

    params->num_functions = 4;
    params->num_terminals = 2;
    params->ops = MALLOC(6, sizeof(enum gp_operator));
    params->ops[0] = ADD;
    params->ops[1] = SUB;
    params->ops[2] = MUL;
    params->ops[3] = SIN;
    params->ops[4] = ERC;
    params->ops[5] = VAR;

    params->standardise = true;

    params->coef_op = true;
    params->learning_rate = 0.01;
    params->learning_epochs = 3;

    params->mutation_sigma = 0.1;

    params->timeout = 0; /* no timeout */

    return params;
}

void gp_free_parameters(struct gp_parameters *params)
{
    if (params == NULL) return;
    free(params->ops);
    free(params);
}

static int cmp_op(const void *a, const void *b)
{
    return *((const enum gp_operator *)a) - *((const enum gp_operator *)b);
}

void gp_init_function_set(struct gp_parameters *params, const char *op_list)
{
    static uint8_t choices = 16;
    static char *names[]          = { "ADD", "SUB", "MUL", "DIV", "SIN", "COS", "EXP", "LOG", "INV", "AQT", "PDIV", "PEXP", "PLOG", "PINV", "ERC", "VAR" };
    static enum gp_operator ops[] = {  ADD,   SUB,   MUL,   DIV,   SIN,   COS,   EXP,   LOG,   INV,   AQT,   PDIV,   PEXP,   PLOG,   PINV,   ERC,   VAR  };
    static uint8_t arity[]        = {    2,     2,     2,     2,     2,     1,     1,     1,     1,     2,      2,      1,      1,      1,     0,     0  };

    char *buf = strdup(op_list);
    char *tok;
    bool var_defined = false, found;
    uint8_t n_op = 0, i;

    free(params->ops);
    params->ops = MALLOC(choices, sizeof(enum gp_operator));
    params->num_functions = params->num_terminals = 0;

    /* function and terminals are provided in a comma-separated list - we don't check for duplicates! */
    for (tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        found = false;
        for (i = 0; i < choices; ++i) {
            if (strcmp(tok, names[i]) == 0) {
                if (arity[i] > 0) {
                    params->num_functions++;
                } else {
                    params->num_terminals++;
                }
                params->ops[n_op++] = ops[i];
                var_defined = var_defined || (strcmp(tok, "VAR") == 0);
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "%s:%d - WARNING: Unknown operator %s\n", __FILE__, __LINE__, tok);
        }
    }

    free(buf);

    if (!var_defined) {
        params->num_terminals++;
        params->ops[n_op++] = VAR;
    }

    qsort(params->ops, n_op, sizeof(enum gp_operator), cmp_op);
}

















static void prvar(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "X[:,%d]", tree->i);
}
static void prerc(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "%.*e", DECIMAL_DIG, tree->f);
}

static void pradd(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, " + ");
    tree->arg1->pr(tree->arg1, out);
    fprintf(out, ")");
}
static void prsub(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, " - ");
    tree->arg1->pr(tree->arg1, out);
    fprintf(out, ")");
}
static void prmul(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, " * ");
    tree->arg1->pr(tree->arg1, out);
    fprintf(out, ")");
}
static void prdiv(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, " / ");
    tree->arg1->pr(tree->arg1, out);
    fprintf(out, ")");
}
static void praqt(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, " / np.sqrt(1.0 + ");
    tree->arg1->pr(tree->arg1, out);
    fprintf(out, "**2))");
}

static void prsin(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "np.sin(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, ")");
}
static void prcos(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "np.cos(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, ")");
}
static void prexp(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "np.exp(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, ")");
}
static void prlog(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "np.log(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, ")");
}
static void prinv(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "(1.0 / ");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, ")");
}

static void prpdiv(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, " %%div%% ");
    tree->arg1->pr(tree->arg1, out);
    fprintf(out, ")");
}
static void prpexp(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "pexp(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, ")");
}
static void prplog(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "plog(");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, ")");
}
static void prpinv(struct parse_tree *tree, FILE *out)
{
    fprintf(out, "(1.0 %%div%% ");
    tree->arg0->pr(tree->arg0, out);
    fprintf(out, ")");
}



static double opvar(struct parse_tree *tree, double *data)
{
    return tree->f = data[tree->i];
}
static double operc(struct parse_tree *tree, double *data __attribute__((unused)))
{
    return tree->f;
}

static double opadd(struct parse_tree *tree, double *data)
{
    return tree->f = (tree->arg0->op(tree->arg0, data) + tree->arg1->op(tree->arg1, data));
}
static double opsub(struct parse_tree *tree, double *data)
{
    return tree->f = (tree->arg0->op(tree->arg0, data) - tree->arg1->op(tree->arg1, data));
}
static double opmul(struct parse_tree *tree, double *data)
{
    return tree->f = (tree->arg0->op(tree->arg0, data) * tree->arg1->op(tree->arg1, data));
}
static double opdiv(struct parse_tree *tree, double *data)
{
    return tree->f = (tree->arg0->op(tree->arg0, data) / tree->arg1->op(tree->arg1, data));
}
static double opaqt(struct parse_tree *tree, double *data)
{
    double b;
    b = tree->arg1->op(tree->arg1, data);
    return tree->f = (tree->arg0->op(tree->arg0, data) / sqrt(1 + b*b));
}

static double opsin(struct parse_tree *tree, double *data)
{
    return tree->f = sin(tree->arg0->op(tree->arg0, data));
}
static double opcos(struct parse_tree *tree, double *data)
{
    return tree->f = cos(tree->arg0->op(tree->arg0, data));
}
static double opexp(struct parse_tree *tree, double *data)
{
    return tree->f = exp(tree->arg0->op(tree->arg0, data));
}
static double oplog(struct parse_tree *tree, double *data)
{
    return tree->f = log(tree->arg0->op(tree->arg0, data));
}
static double opinv(struct parse_tree *tree, double *data)
{
    return tree->f = (1 / tree->arg0->op(tree->arg0, data));
}

static double oppdiv(struct parse_tree *tree, double *data)
{
    double b;
    b = tree->arg1->op(tree->arg1, data);
    return tree->f = (fabs(b) < SMALL_LIMIT) ? 1 : (tree->arg0->op(tree->arg0, data) / b);
}
static double oppexp(struct parse_tree *tree, double *data)
{
    double a;
    a = tree->arg0->op(tree->arg0, data);
    return (fabs(a) < EXP_LIMIT) ? exp(a) : (a < 0) ? 0 : BIG_NUMBER;
}
static double opplog(struct parse_tree *tree, double *data)
{
    double a;
    a = tree->arg0->op(tree->arg0, data);
    return tree->f = (fabs(a) < SMALL_LIMIT) ? 0 : log(fabs(a));
}
static double oppinv(struct parse_tree *tree, double *data)
{
    double a;
    a = tree->arg0->op(tree->arg0, data);
    return tree->f = (fabs(a) < SMALL_LIMIT) ? 1 : (1 / a);
}

static uint16_t advar(struct parse_tree *w __attribute__((unused)), double wbar __attribute__((unused)), double *wstar __attribute__((unused)), uint16_t widx)
{
    return widx;
}
static uint16_t aderc(struct parse_tree *w __attribute__((unused)), double wbar, double *wstar, uint16_t widx)
{
    wstar[widx] += wbar;
    return widx + 1;
}

static uint16_t adadd(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of both sides is wbar * 1 */
    widx = w->arg0->ad(w->arg0, wbar, wstar, widx);
    widx = w->arg1->ad(w->arg1, wbar, wstar, widx);
    return widx;
}
static uint16_t adsub(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar * 1 */
    /* derivative of arg1 is wbar * -1 */
    widx = w->arg0->ad(w->arg0, wbar, wstar, widx);
    widx = w->arg1->ad(w->arg1, -wbar, wstar, widx);
    return widx;
}
static uint16_t admul(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar * f(arg1) */
    /* derivative of arg1 is wbar * f(arg0) */
    widx = w->arg0->ad(w->arg0, wbar * w->arg1->f, wstar, widx);
    widx = w->arg1->ad(w->arg1, wbar * w->arg0->f, wstar, widx);
    return widx;
}
static uint16_t addiv(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar * 1 / f(arg1) */
    /* derivative of arg1 is wbar * -f(arg0) / f(arg1)^2  */
    widx = w->arg0->ad(w->arg0, wbar / w->arg1->f, wstar, widx);
    widx = w->arg1->ad(w->arg1, -wbar * w->arg0->f / (w->arg1->f * w->arg1->f), wstar, widx);
    return widx;
}
static uint16_t adaqt(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar * 1 / sqrt(f(arg1)^2 + 1) */
    /* derivative of arg1 is wbar * -f(arg0)*f(arg1) / (f(arg1)^2 + 1)^3/2  */
    widx = w->arg0->ad(w->arg0, wbar / sqrt(w->arg1->f * w->arg1->f + 1), wstar, widx);
    widx = w->arg1->ad(w->arg1, -wbar * w->arg0->f * w->arg1->f / sqrt((1 + w->arg1->f * w->arg1->f) * (1 + w->arg1->f * w->arg1->f) * (1 + w->arg1->f * w->arg1->f)), wstar, widx);
    return widx;
}

static uint16_t adsin(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar * cos(f(arg0)) */
    widx = w->arg0->ad(w->arg0, wbar * cos(w->arg0->f), wstar, widx);
    return widx;
}
static uint16_t adcos(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar * -sin(f(arg0)) */
    widx = w->arg0->ad(w->arg0, -wbar * sin(w->arg0->f), wstar, widx);
    return widx;
}
static uint16_t adexp(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar * exp(f(arg0)) */
    widx = w->arg0->ad(w->arg0, wbar * exp(w->arg0->f), wstar, widx);
    return widx;
}
static uint16_t adlog(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar *  1 / f(arg0) */
    widx = w->arg0->ad(w->arg0, wbar / w->arg0->f, wstar, widx);
    return widx;
}
static uint16_t adinv(struct parse_tree *w, double wbar, double *wstar, uint16_t widx)
{
    /* derivative of arg0 is wbar * -1 / f(arg0)^2  */
    widx = w->arg0->ad(w->arg0, -wbar * w->arg0->f / (w->arg0->f * w->arg0->f), wstar, widx);
    return widx;
}

static uint16_t aderr(struct parse_tree *w __attribute__((unused)),
                      double wbar __attribute__((unused)),
                      double *wstar __attribute__((unused)),
                      uint16_t widx __attribute__((unused)))
{
    fprintf(stderr, "%s:%d = Attempt to use autodiff with protected function - don't!\n", __FILE__, __LINE__);
    exit(EXIT_FAILURE);
    return -1;
}










static void destroy_tree(struct parse_tree *tree)
{
    if (tree == NULL) return;

    destroy_tree(tree->arg0);
    destroy_tree(tree->arg1);
    free(tree);
}



static struct parse_tree *copy_tree(struct parse_tree *tree)
{
    if (tree == NULL) return NULL;

    struct parse_tree *copy = MALLOC(1, sizeof(struct parse_tree));
    copy->op = tree->op;
    copy->ad = tree->ad;
    copy->pr = tree->pr;
    copy->size = tree->size;
    copy->depth = tree->depth;
    copy->parent = NULL;

    copy->arg0 = copy_tree(tree->arg0);
    if (copy->arg0) copy->arg0->parent = copy;

    copy->arg1 = copy_tree(tree->arg1);
    if (copy->arg1) copy->arg1->parent = copy;

    copy->f   = (tree->op == operc) ? tree->f   : NAN;
    copy->i   = (tree->op == opvar) ? tree->i   : -1;

    return copy;
}


static double eval(struct gp *ind, double **X, double *t, uint32_t *subset, uint32_t n_samples)
{
    double s = 0;
    for (uint32_t i = 0; i < n_samples; ++i) {
        double ti = t[subset[i]];
        double yi = ind->tree->op(ind->tree, X[subset[i]]);
        double r = (ti - yi);

        s += ((r * r) - s) / (i + 1);
    }

    return s;
}


static uint16_t count_coefficients(struct parse_tree *n)
{
    if (n->op == operc) return 1;
    if (n->op == opvar) return 0;

    if (n->arg1) {
        return count_coefficients(n->arg0) + count_coefficients(n->arg1);
    } else {
        return count_coefficients(n->arg0);
    }
}


static uint16_t read_coefficients(struct parse_tree *n, struct parse_tree **C, uint16_t idx)
{
    if (n->op == operc) {
        C[idx++] = n;
        return idx;
    }

    if (n->arg0) idx = read_coefficients(n->arg0, C, idx);
    if (n->arg1) idx = read_coefficients(n->arg1, C, idx);
    return idx;
}


static void coef_op(struct gp *ind,
                    double **X, double *t, uint32_t *subset, uint32_t n_samples,
                    struct gp_parameters *params, rng_state state)
{
    uint16_t n_opt = count_coefficients(ind->tree);

    if (n_opt == 0) return;

    uint32_t *shuff = MALLOC(n_samples, sizeof(uint32_t));
    double   *grad  = MALLOC(n_opt, sizeof(double));
    struct parse_tree **C = MALLOC(n_opt, sizeof(struct parse_tree *));

    read_coefficients(ind->tree, C, 0);
    double eta = params->learning_rate < 0 ? (-params->learning_rate / n_samples) : params->learning_rate;

    for (uint16_t e = 0; e < params->learning_epochs; ++e) {
        for (uint32_t i = 0; i < n_samples; ++i) {
            uint32_t j = next_rnd(state) * i;
            if (i != j) shuff[i] = shuff[j];
            shuff[j] = subset[i];
        }

        for (uint32_t i = 0; i < n_samples; ++i) {
            double ti = t[shuff[i]];
            double yi = ind->tree->op(ind->tree, X[shuff[i]]);

            for (uint16_t j = 0; j < n_opt; ++j) grad[j] = 0;
            ind->tree->ad(ind->tree, -2 * (ti - yi), grad, 0);

            for (uint16_t j = 0; j < n_opt; ++j) C[j]->f -= eta * grad[j];
        }
    }

    free(C);
    free(grad);
    free(shuff);
}



static struct gp *new_individual()
{
    struct gp *ind = MALLOC(1, sizeof(struct gp));

    ind->p    = -1;

    ind->tree = NULL;
    ind->data = NULL;

    ind->s = ind->mu = NULL;

    return ind;
}

static struct parse_tree *init_tree(struct gp_parameters *params,
                                    uint8_t depth, uint16_t min_depth, uint16_t max_depth,
                                    uint16_t J, rng_state state)
{
    // enum gp_operator    {   ADD,   SUB,   MUL,   DIV,   AQT,  PDIV,    SIN,   COS,   EXP,   LOG,   INV,   PEXP,   PLOG,   PINV,   ERC,   VAR };
    static nodeop nops[] = { opadd, opsub, opmul, opdiv, opaqt, oppdiv, opsin, opcos, opexp, oplog, opinv, oppexp, opplog, oppinv, operc, opvar };
    static nodead nads[] = { adadd, adsub, admul, addiv, adaqt,  addiv, adsin, adcos, adexp, adlog, adinv,  aderr,  aderr,  aderr, aderc, advar };
    static nodepr nprs[] = { pradd, prsub, prmul, prdiv, praqt, prpdiv, prsin, prcos, prexp, prlog, prinv, prpexp, prplog, prpinv, prerc, prvar };

    struct parse_tree *ret = MALLOC(1, sizeof(struct parse_tree));
    ret->parent = ret->arg0 = ret->arg1 = NULL;
    ret->ad = aderr;
    ret->size = 1;
    ret->depth = 0;
    ret->f = NAN;
    ret->i = -1;

    double term_prob = (params->num_terminals) / (double)(params->num_terminals + params->num_functions);

    bool pick_function;
    if (depth == max_depth || next_rnd(state) < term_prob) {
        pick_function = depth < min_depth;
    } else {
        pick_function = true;
    }

    enum gp_operator op;
    if (pick_function) {
        op = params->ops[(uint16_t)(next_rnd(state) * params->num_functions)];
        ret->arg0 = init_tree(params, depth + 1, min_depth, max_depth, J, state);
        ret->arg0->parent = ret;
        ret->size += ret->arg0->size;
        ret->depth = ret->arg0->depth + 1;
        if (BINARY_OPERATOR(op)) {
            ret->arg1 = init_tree(params, depth + 1, min_depth, max_depth, J, state);
            ret->arg1->parent = ret;
            ret->size += ret->arg1->size;
            if (ret->arg1->depth > ret->arg0->depth) ret->depth = ret->arg1->depth + 1;
        }
    } else {
        op = params->ops[params->num_functions + (uint16_t)(next_rnd(state) * params->num_terminals)];
        if (op == ERC) {
            ret->f = next_rnd_gauss(0, params->mutation_sigma, state);
        } else {
            ret->i = (int32_t)(J * next_rnd(state));
        }
    }
    ret->op = nops[op];
    ret->ad = nads[op];
    ret->pr = nprs[op];

    return ret;
}



static void update_tree_stats(struct parse_tree *tree)
{
    if (tree == NULL) return;

    tree->size = 1;
    tree->depth = 0;

    if (tree->arg0) {
        tree->arg0->parent = tree;
        tree->size += tree->arg0->size;
        tree->depth = tree->arg0->depth + 1;
    }

    if (tree->arg1) {
        tree->arg1->parent = tree;
        tree->size += tree->arg1->size;
        if (tree->arg1->depth > tree->arg0->depth) tree->depth = tree->arg1->depth + 1;
    }

    update_tree_stats(tree->parent);
}



static struct parse_tree *pick_subtree(struct parse_tree *tree, uint16_t target)
{
    if (target == 0) return tree;

    if (target <= tree->arg0->size) {
        return pick_subtree(tree->arg0, target - 1);
    } else {
        return pick_subtree(tree->arg1, target - tree->arg0->size - 1);
    }

    /* shouldn't get here */
    fprintf(stderr, "%s:%d - Shouldn't have ended up here\n", __FILE__, __LINE__);
    return NULL;
}



static void subtree_mutation(struct parse_tree *p, struct parse_tree **o_ptr,
                             uint16_t J,
                             struct gp_parameters *params, rng_state state)
{
    struct parse_tree *o = copy_tree(p);

    /* Koza's 90-10 rule for node picking */
    struct parse_tree *mpo;
    do {
        /* Koza's 90-10 rule for node picking */
        if ((next_rnd(state) < 0.9) && (o->size > 1)) {
            do { mpo = pick_subtree(o, next_rnd(state) * o->size); } while (mpo->op == operc || mpo->op == opvar);
        } else {
            mpo = pick_subtree(o, next_rnd(state) * o->size);
        }
    } while (params->max_tree_nodes > 0 && (o->size - mpo->size) >= params->max_tree_nodes);
    struct parse_tree *pmpo = mpo->parent;

    /* ensure that the mutant subtree does not invalidate any tree size limits */
    uint16_t available = params->max_tree_nodes - (o->size - mpo->size);
    uint16_t mut_depth = (uint16_t)(log2(available + 1) - 1); /* assumes all internal nodes have arity=2 */
    if (mut_depth > 4) mut_depth = 4;

    struct parse_tree *s = init_tree(params, 0, 0, mut_depth, J, state);

    if (pmpo) {
        /* mutation within tree */
        if (pmpo->arg0 == mpo) {
            pmpo->arg0 = s;
        } else if (pmpo->arg1 == mpo)  {
            pmpo->arg1 = s;
        }
        s->parent = pmpo;
    } else {
        /* mutation at root */
        o = s;
    }

    destroy_tree(mpo);
    update_tree_stats(s);

    /* check the depths here */
    if ((params->max_tree_depth > 0) && (o->depth > params->max_tree_depth)) {
        destroy_tree(o);
        o = copy_tree(p);
    }

    *o_ptr = o;
}



static void point_mutation(struct parse_tree *p, struct parse_tree **o_ptr,
                           uint16_t J,
                           struct gp_parameters *params, rng_state state)
{
    // enum gp_operator    {   ADD,   SUB,   MUL,   DIV,   AQT,  PDIV,    SIN,   COS,   EXP,   LOG,   INV,   PEXP,   PLOG,   PINV,   ERC,   VAR };
    static nodeop nops[] = { opadd, opsub, opmul, opdiv, opaqt, oppdiv, opsin, opcos, opexp, oplog, opinv, oppexp, opplog, oppinv, operc, opvar };
    static nodead nads[] = { adadd, adsub, admul, addiv, adaqt,  addiv, adsin, adcos, adexp, adlog, adinv,  aderr,  aderr,  aderr, aderc, advar };
    static nodepr nprs[] = { pradd, prsub, prmul, prdiv, praqt, prpdiv, prsin, prcos, prexp, prlog, prinv, prpexp, prplog, prpinv, prerc, prvar };

    struct parse_tree *o = copy_tree(p);
    *o_ptr = o;

    /* Koza's 90-10 rule for node picking */
    if (o->size == 1) return;

    struct parse_tree *mpo;
    if ((next_rnd(state) < 0.9) && (o->size > 1)) {
        do { mpo = pick_subtree(o, next_rnd(state) * o->size); } while (mpo->op == operc || mpo->op == opvar);
    } else {
        mpo = pick_subtree(o, next_rnd(state) * o->size);
    }

    enum gp_operator op;
    if (mpo->arg0 && mpo->arg1) {
        do { op = params->ops[(uint8_t)(params->num_functions * next_rnd(state))]; } while (UNARY_OPERATOR(op));
    } else if (mpo->arg0) {
        do { op = params->ops[(uint8_t)(params->num_functions * next_rnd(state))]; } while (BINARY_OPERATOR(op));
    } else {
        op = params->ops[params->num_functions + (uint8_t)(next_rnd(state) * params->num_terminals)];
    }

    if (op == ERC) {
        mpo->f = next_rnd_gauss(0, params->mutation_sigma, state);
        mpo->i = -1;
    } else if (op == VAR) {
        mpo->f = NAN;
        mpo->i = (int32_t)(J * next_rnd(state));
    }
    mpo->op = nops[op];
    mpo->ad = nads[op];
    mpo->pr = nprs[op];
}



static void crossover(struct parse_tree *m, struct parse_tree *f,
                      struct parse_tree **o_ptr,
                      struct gp_parameters *params, rng_state state)
{
    struct parse_tree *d = copy_tree(m);
    struct parse_tree *s = copy_tree(f);

    /* Koza's 90-10 rule for node picking */
    struct parse_tree *cpd;
    do {
        if ((next_rnd(state) < 0.9) && (d->size > 1)) {
            do { cpd = pick_subtree(d, next_rnd(state) * d->size); } while (cpd->op == operc || cpd->op == opvar);
        } else {
            cpd = pick_subtree(d, next_rnd(state) * d->size);
        }
    } while (params->max_tree_nodes > 0 && (d->size - cpd->size) >= params->max_tree_nodes);
    struct parse_tree *pcpd = cpd->parent;

    /* Koza's 90-10 rule for node picking */
    struct parse_tree *cps;
    do { /* pick a crossover point in parent 2 that will not invalidate tree sizes */
        /* Koza's 90-10 rule for node picking */
        if ((next_rnd(state) < 0.9) && (s->size > 1)) {
            do { cps = pick_subtree(s, next_rnd(state) * s->size); } while (cps->op == operc || cps->op == opvar);
        } else {
            cps = pick_subtree(s, next_rnd(state) * s->size);
        }
    } while (params->max_tree_nodes > 0 && (d->size - cpd->size + cps->size) > params->max_tree_nodes);
    struct parse_tree *pcps = cps->parent;


    if (pcpd && pcps) {
        /* crossover at non-root in both parents */
        if (pcpd->arg0 == cpd) {
            pcpd->arg0 = cps;
        } else {
            pcpd->arg1 = cps;
        }
        if (pcps->arg0 == cps) {
            pcps->arg0 = cpd;
        } else {
            pcps->arg1 = cpd;
        }
    } else if (pcpd) {
        /* crossover at cps is at root, cpd non-root */
        if (pcpd->arg0 == cpd) {
            pcpd->arg0 = cps;
        } else if (pcpd->arg1 == cpd)  {
            pcpd->arg1 = cps;
        }
        s = cpd;
    } else if (pcps) {
        d = cps;
        if (pcps->arg0 == cps) {
            pcps->arg0 = cpd;
        } else if (pcps->arg1 == cps)  {
            pcps->arg1 = cpd;
        }
    } else {
        struct parse_tree *t = d;
        d = s;
        s = t;
    }
    cpd->parent = pcps;
    cps->parent = pcpd;

    update_tree_stats(cps);

    /* check the depths here */
    if (params->max_tree_depth > 0 && d->depth > params->max_tree_depth) {
        destroy_tree(d);
        d = copy_tree(m);
    }

    *o_ptr = d;
    destroy_tree(s);
}



struct ranking_info {
    uint16_t i;
    double fitness;
    uint16_t size;
};

static int cmp_fit(double a, double b)
{
    if (sane(a) && sane(b)) {
        return (a < b) ? -1 : (a > b) ? 1 : 0;
    } else if (sane(a)) {
        return -1;
    } else if (sane(b)) {
        return  1;
    } else {
        return  0;
    }
}

static int cmp_rank(const void * a_ptr, const void * b_ptr)
{
    const struct ranking_info *a = a_ptr;
    const struct ranking_info *b = b_ptr;

    int cfit = cmp_fit(a->fitness, b->fitness);
    if (cfit == 0) {
        if (a->size < b->size) return -1;
        if (a->size > b->size) return 1;
        return 0;
    } else {
        return cfit;
    }
}


static uint16_t tournament(double *fit, int32_t last, struct gp_parameters *params, rng_state state)
{

    uint16_t K = params->tournament_size < 1 ? (params->tournament_size * params->pop_size) : params->tournament_size;

    uint16_t a, b;
    do { a = params->pop_size * next_rnd(state); } while (a == last);
    for (uint16_t k = 1; k < K; ++k) {
        do { b = params->pop_size * next_rnd(state); } while (b == a || b == last);
        if (cmp_fit(fit[b], fit[a]) < 0) a = b;
    }

    return a;
}

static void setup_data(double **raw_X, double *raw_t, uint32_t n_samples, uint16_t n_feat,
                       bool standardise,
                       double ***X_ptr, double **t_ptr, double **mu_ptr, double **s_ptr)
{
    double delta;

    /* then copy the current training instances into our working set */
    double **X = MALLOC(n_samples, sizeof(double *));
    double  *t = MALLOC(n_samples, sizeof(double));
    X[0] = MALLOC(n_samples * n_feat, sizeof(double));
    for (uint32_t i = 0; i < n_samples; ++i) {
        X[i] = X[0] + i * n_feat;
        for (uint32_t j = 0; j < n_feat; ++j) {
            X[i][j] = raw_X[j][i];
        }
        t[i] = raw_t[i];
    }

    /* then determine the training mean and standard deviation */
    double *mu = MALLOC(n_feat + 1, sizeof(double));
    double *s  = MALLOC(n_feat + 1, sizeof(double));
    for (uint16_t j = 0; j <= n_feat; ++j) mu[j] = s[j] = 0;

    for (uint32_t i = 0; i < n_samples; ++i) {
        for (uint16_t j = 0; j < n_feat; ++j) {
            delta = X[i][j] - mu[j];
            mu[j] += delta / (i + 1);
            s[j]  += delta * (X[i][j] - mu[j]);
        }

        delta = t[i] - mu[n_feat];
        mu[n_feat] += delta / (i + 1);
        s[n_feat]  += delta * (t[i] - mu[n_feat]);
    }
    for (uint16_t j = 0; j <= n_feat; ++j) s[j] = sqrt(s[j] / (n_samples - 1));

    /* standardise the input variables, if required, or reset mean and standard deviations */
    if (standardise) {
        for (uint32_t i = 0; i < n_samples; ++i) {
            for (uint16_t j = 0; j < n_feat; ++j) X[i][j] = (X[i][j] - mu[j]) / s[j];
            t[i] = (t[i] - mu[n_feat]) / s[n_feat];
        }
    } else {
        for (uint16_t j = 0; j <= n_feat; ++j) {
            mu[j] = 0;
            s[j] = 1;
        }
    }

    *X_ptr = X;
    *t_ptr = t;
    *mu_ptr = mu;
    *s_ptr = s;
}

static void sample_validation(struct gp_parameters *params, double pvalid, uint32_t n_samples,
                              uint32_t *train, uint32_t *n_train_ptr,
                              uint32_t *valid, uint32_t *n_valid_ptr)
{
    if (pvalid < 0) {
        fprintf(stderr, "%s:%d - WARNING - supplied value of %f for p, trimming to zero and returning NULL\n", __FILE__, __LINE__, pvalid);
        pvalid = 0;
    } else if (pvalid > 1) {
        fprintf(stderr, "%s:%d - WARNING - supplied value of %f for p, truncating to 1 and returning all\n", __FILE__, __LINE__, pvalid);
        pvalid = 1;
    }

    double   p = 1 - pvalid;
    uint32_t m = (uint32_t)(p * n_samples);
    uint32_t n = n_samples;

    uint32_t i, j, k;
    for (k = 0, j = 0, i = 0; i < n && j < m; ++i) {
        if (next_rnd(params->random_state) < ((double)(m - j) / (double)(n - i))) {
            train[j++] = i;
        } else {
            valid[k++] = i;
        }
    }

    *n_train_ptr = j;
    *n_valid_ptr = k;
}

static double validation_elitism(struct gp **pop, uint16_t pop_size,
                                 struct gp *ret, double ret_fit,
                                 double **X, double *t, uint32_t *valid, uint32_t n_valid,
                                 double *vfit, uint16_t best_train, double best_fitness)
{
    if (n_valid == 0) {
        destroy_tree(ret->tree);
        ret->tree = copy_tree(pop[best_train]->tree);
        return best_fitness;
    } else {
#pragma omp parallel for
        for (uint16_t i = 0; i < pop_size; ++i) {
            vfit[i] = eval(pop[i], X, t, valid, n_valid);
        }

        uint16_t vbest = 0;
        for (uint16_t i = 1; i < pop_size; ++i) {
            if (cmp_fit(vfit[i], vfit[vbest]) < 0) vbest = i;
        }

        if (cmp_fit(vfit[vbest], ret_fit) < 0) {
            ret_fit = vfit[vbest];
            destroy_tree(ret->tree);
            ret->tree = copy_tree(pop[vbest]->tree);
        }
        return ret_fit;
    }
}

static rng_state *seed_rngs(struct gp_parameters *params)
{
    rng_state *states = MALLOC(params->pop_size, sizeof(rng_state));

    for (uint16_t i = 0; i < params->pop_size; ++i) spawn_rng(params->random_state, states + i);

    return states;
}

static void process_signal(int signum)
{
    if (signum == SIGALRM) interrupted = true;
}

struct gp *gp_evolve(struct gp_parameters *params, double **raw_X, double *raw_t, uint32_t n_samples, uint16_t n_feat)
{
    interrupted = false;
    signal(SIGALRM, process_signal);
    alarm(params->timeout);

    rng_state *states = seed_rngs(params);

    uint32_t *train = MALLOC(n_samples, sizeof(uint32_t)), n_train=0, *valid = MALLOC(n_samples, sizeof(uint32_t)), n_valid = 0;

    double **X, *t, *mu, *s;

    void *swp;

    setup_data(raw_X, raw_t, n_samples, n_feat, params->standardise, &X, &t, &mu, &s);

    sample_validation(params, params->validation_prop, n_samples, train, &n_train, valid, &n_valid);

    uint8_t init_min_depth = params->min_tree_init;
    uint8_t init_max_depth = params->max_tree_init;
    if (params->max_tree_depth > 0) {
        if (params->max_tree_depth < init_min_depth) init_min_depth = params->max_tree_depth;
        if (params->max_tree_depth < init_max_depth) init_max_depth = params->max_tree_depth;
    }

    uint16_t elitism_size = (params->elitism_rate < 1) ? (params->elitism_rate * params->pop_size) : params->elitism_rate;

    double *data = MALLOC(n_feat, sizeof(double));

    struct ranking_info *ranking = MALLOC(params->pop_size, sizeof(struct ranking_info));
    double *pfit = MALLOC(params->pop_size, sizeof(double));
    double *gfit = MALLOC(params->pop_size, sizeof(double));
    double *vfit = MALLOC(params->pop_size, sizeof(double));

    struct gp **pop = MALLOC(params->pop_size, sizeof(struct gp *));
    struct gp **gen = MALLOC(params->pop_size, sizeof(struct gp *));

    /* create a placeholder for the best evolved individual */
    struct gp *ret = new_individual();
    ret->data  = data;
    ret->mu    = params->standardise ? mu : NULL;
    ret->s     = params->standardise ? s : NULL;
    ret->p     = n_feat;
    ret->tree  = NULL;

    double ret_fit = NAN;

    /* init individuals */
#pragma omp parallel for
    for (uint16_t i = 0; i < params->pop_size; ++i) {
        pop[i] = new_individual();
        gen[i] = new_individual();

        gen[i]->data  = pop[i]->data  = data;
        gen[i]->mu    = pop[i]->mu    = mu;
        gen[i]->s     = pop[i]->s     = s;
        gen[i]->p     = pop[i]->p     = n_feat;
        gen[i]->tree  = pop[i]->tree  = NULL;

        uint8_t depth = init_min_depth + (i % (1 + init_max_depth - init_min_depth));

        if (i < (params->pop_size / 2)) {
            pop[i]->tree = init_tree(params, 0, init_min_depth, depth, n_feat, states[i]);
        } else {
            pop[i]->tree = init_tree(params, 0, depth, depth, n_feat, states[i]);
        }
        if (params->coef_op) coef_op(pop[i], X, t, train, n_train, params, states[i]);

        pfit[i]            = eval(pop[i], X, t, train, n_train);
        ranking[i].i       = i;
        ranking[i].fitness = pfit[i];
        ranking[i].size    = pop[i]->tree->size;
    }

    qsort(ranking, params->pop_size, sizeof(struct ranking_info), cmp_rank);
    ret_fit = validation_elitism(pop, params->pop_size, ret, ret_fit, X, t, valid, n_valid, vfit, ranking[0].i, ranking[0].fitness);

    /* generations */
    for (uint16_t g = 1; !interrupted && g <= params->generations; ++g) {
        for (uint16_t i = 0; i < elitism_size; ++i) {
            destroy_tree(gen[i]->tree);
            gen[i]->tree  = copy_tree(pop[ranking[i].i]->tree);
            gfit[i]       = ranking[i].fitness;
            ranking[i].i  = i;
        }

#pragma omp parallel for
        for (uint16_t i = elitism_size; i < params->pop_size; ++i) {
            if (interrupted) continue;

            destroy_tree(gen[i]->tree);

            uint16_t m = tournament(pfit, -1, params, states[i]);

            double p = next_rnd(states[i]);
            if (p < params->crossover_rate) {
                uint16_t f = tournament(pfit, m, params, states[i]);
                crossover(pop[m]->tree, pop[f]->tree, &(gen[i]->tree), params, states[i]);
            } else if (p < (params->crossover_rate + params->point_mutation_rate)) {
                point_mutation(pop[m]->tree, &(gen[i]->tree), n_feat, params, states[i]);
            } else if (p < (params->crossover_rate + params->point_mutation_rate + params->sub_mutation_rate)) {
                subtree_mutation(pop[m]->tree, &(gen[i]->tree), n_feat, params, states[i]);
            } else {
                gen[i]->tree = copy_tree(pop[m]->tree);
            }
            if (params->coef_op) coef_op(gen[i], X, t, train, n_train, params, states[i]);

            gfit[i]            = eval(gen[i], X, t, train, n_train);
            ranking[i].i       = i;
            ranking[i].fitness = gfit[i];
            ranking[i].size    = gen[i]->tree->size;
        }

        swp = pop;
        pop = gen;
        gen = swp;

        swp  = pfit;
        pfit = gfit;
        gfit = swp;

        qsort(ranking, params->pop_size, sizeof(struct ranking_info), cmp_rank);
        ret_fit = validation_elitism(pop, params->pop_size, ret, ret_fit, X, t, valid, n_valid, vfit, ranking[0].i, ranking[0].fitness);
    }


    /* cleanup */
    for (uint16_t i = 0; i < params->pop_size; ++i) {
        gen[i]->s    = pop[i]->s = NULL;
        gen[i]->mu   = pop[i]->mu = NULL;
        gen[i]->data = pop[i]->data = NULL;

        gp_free(pop[i]);
        gp_free(gen[i]);
    }

    if (!params->standardise) {
        free(mu);
        free(s);
    }

    free(pop);
    free(gen);

    free(ranking);
    free(vfit);
    free(pfit);
    free(gfit);

    free(t);
    free(X[0]);
    free(X);

    free(valid);
    free(train);

    return ret;
}

void gp_free(struct gp *ind)
{
    if (ind == NULL) return;

    destroy_tree(ind->tree);

    free(ind->s);

    free(ind->mu);

    free(ind->data);

    free(ind);
}

void gp_print(FILE *out, struct gp *ind)
{
    if (ind->mu) {
        fprintf(out, "np.array([ %.*e", DECIMAL_DIG, ind->mu[0]);
        for (uint16_t j = 1; j < ind->p; ++j) fprintf(out, ", %.*e", DECIMAL_DIG, ind->mu[j]);
        fprintf(out, " ]);np.array([ %.*e", DECIMAL_DIG, ind->s[0]);
        for (uint16_t j = 1; j < ind->p; ++j) fprintf(out, ", %.*e", DECIMAL_DIG, ind->s[j]);
        fprintf(out, " ]); %.*e + %.*e * (", DECIMAL_DIG, ind->mu[ind->p], DECIMAL_DIG, ind->s[ind->p]);
    } else {
        fprintf(out, "float(0);float(1);");
    }

    ind->tree->pr(ind->tree, out);

    if (ind->mu) fprintf(out, ")");
}


void gp_predict(struct gp *ind, double **X, uint32_t n_samples, double *y)
{
    if (ind->mu) { /* using standardisation */
        for (uint32_t i = 0; i < n_samples; ++i) {
            for (uint16_t j = 0; j < ind->p; ++j) ind->data[j] = (X[j][i] - ind->mu[j]) / ind->s[j];
            y[i] = ind->mu[ind->p] + ind->s[ind->p] * ind->tree->op(ind->tree, ind->data);
        }
    } else { /* not using standardisation */
        for (uint32_t i = 0; i < n_samples; ++i) {
            for (uint16_t j = 0; j < ind->p; ++j) ind->data[j] = X[j][i];
            y[i] = ind->tree->op(ind->tree, ind->data);
        }
    }
}

uint16_t gp_size(struct gp *ind)
{
    return ind->tree->size;
}
