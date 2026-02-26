/*
 * wl_tm_api.c -- C glue layer for Tsetlin Machine via TMU
 *
 * Wraps TMU's ClauseBank.c / WeightBank.c with:
 *   - training loop (epochs, multiclass, regression)
 *   - binarization (quantile-based thresholds from continuous features)
 *   - TM01 binary serialization format
 *
 * Upstream: TMU v0.8.3 (github.com/cair/tmu), MIT license
 * Author of upstream C core: Ole-Christoffer Granmo
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "ClauseBank.h"
#include "fast_rand.h"

/* ---- error handling ---- */

static char last_error[512] = "";

static void set_error(const char *msg) {
    strncpy(last_error, msg, sizeof(last_error) - 1);
    last_error[sizeof(last_error) - 1] = '\0';
}

/* ---- external PRNG seed (from pcg32_fast.c) ---- */
extern void pcg32_seed(uint64_t seed);

/* ---- model struct ---- */

typedef struct WlTM {
    int n_clauses;
    int n_features;       /* original input features */
    int n_binary;         /* after binarization (thresholded features) */
    int n_classes;
    int state_bits;
    int threshold;        /* T: voting margin */
    int boost_tpf;        /* boost true positive feedback */
    int task;             /* 0 = classification, 1 = regression */
    double s;             /* specificity */
    uint32_t seed;

    /* binarization */
    int n_thresholds_per_feature;
    double *thresholds;       /* [n_features * n_thresholds_per_feature] */
    int *threshold_counts;    /* [n_features] actual count per feature */

    /* automaton state (bit-packed), per class */
    unsigned int *ta_state;   /* [n_classes * n_clauses * la_chunks * state_bits] */
    int n_literals;           /* 2 * n_binary (positive + negated) */
    int la_chunks;            /* ceil(n_literals / 32) */

    /* regression scaling */
    double y_min, y_max;

    /* class labels (classification only) */
    int *class_labels;        /* [n_classes] sorted unique labels */
} WlTM;

/* ---- helpers ---- */

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static int cmp_int(const void *a, const void *b) {
    return (*(const int *)a) - (*(const int *)b);
}

/* Compute quantile-based thresholds for one feature column */
static int compute_thresholds_for_feature(
    const double *col_values, int n,
    int max_thresholds,
    double *out_thresholds
) {
    /* Sort a copy */
    double *sorted = (double *)malloc(n * sizeof(double));
    if (!sorted) return 0;
    memcpy(sorted, col_values, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);

    /* Get unique values */
    int n_unique = 1;
    for (int i = 1; i < n; i++) {
        if (sorted[i] != sorted[i - 1]) n_unique++;
    }

    /* If only 1 unique value, no thresholds needed */
    if (n_unique <= 1) {
        free(sorted);
        return 0;
    }

    /* Compute up to max_thresholds evenly spaced quantile thresholds */
    int n_thresh = max_thresholds;
    if (n_thresh > n_unique - 1) n_thresh = n_unique - 1;

    int count = 0;
    for (int t = 0; t < n_thresh; t++) {
        double q = (double)(t + 1) / (double)(n_thresh + 1);
        double idx_f = q * (n - 1);
        int idx_lo = (int)idx_f;
        int idx_hi = idx_lo + 1;
        if (idx_hi >= n) idx_hi = n - 1;
        double frac = idx_f - idx_lo;
        double val = sorted[idx_lo] * (1.0 - frac) + sorted[idx_hi] * frac;

        /* Avoid duplicate thresholds */
        if (count > 0 && val == out_thresholds[count - 1]) continue;
        out_thresholds[count++] = val;
    }

    free(sorted);
    return count;
}

/* Binarize a single row using thresholds: feature k -> threshold_counts[k] binary features */
static void binarize_row(
    const double *row, int n_features,
    const double *thresholds, const int *threshold_counts,
    int n_thresholds_per_feature,
    unsigned int *Xi, int n_literals, int la_chunks
) {
    /* Xi has la_chunks unsigned ints, bit-packed:
     * first n_binary bits = positive literals (x > threshold)
     * next n_binary bits = negated literals  (x <= threshold)
     */
    memset(Xi, 0, la_chunks * sizeof(unsigned int));

    int bit_pos = 0;
    int n_binary = 0;

    /* Count total binary features */
    for (int f = 0; f < n_features; f++) {
        n_binary += threshold_counts[f];
    }

    /* Set positive literals */
    bit_pos = 0;
    for (int f = 0; f < n_features; f++) {
        const double *f_thresh = &thresholds[f * n_thresholds_per_feature];
        for (int t = 0; t < threshold_counts[f]; t++) {
            if (row[f] > f_thresh[t]) {
                Xi[bit_pos / 32] |= (1U << (bit_pos % 32));
            }
            bit_pos++;
        }
    }

    /* Set negated literals */
    for (int f = 0; f < n_features; f++) {
        const double *f_thresh = &thresholds[f * n_thresholds_per_feature];
        for (int t = 0; t < threshold_counts[f]; t++) {
            if (row[f] <= f_thresh[t]) {
                Xi[bit_pos / 32] |= (1U << (bit_pos % 32));
            }
            bit_pos++;
        }
    }
}

/* Clamp integer to [lo, hi] */
static inline int clamp_int(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

/* ---- exported API ---- */

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

EXPORT const char *wl_tm_get_last_error(void) {
    return last_error;
}

EXPORT WlTM *wl_tm_create(
    int n_clauses, int threshold, double s,
    int state_bits, int boost_tpf,
    int n_thresholds_per_feature, int task,
    unsigned int seed
) {
    if (n_clauses < 2 || n_clauses % 2 != 0) {
        set_error("n_clauses must be even and >= 2");
        return NULL;
    }
    if (threshold < 1) {
        set_error("threshold must be >= 1");
        return NULL;
    }
    if (s <= 0.0) {
        set_error("s must be > 0");
        return NULL;
    }
    if (state_bits < 1 || state_bits > 16) {
        set_error("state_bits must be 1-16");
        return NULL;
    }

    WlTM *tm = (WlTM *)calloc(1, sizeof(WlTM));
    if (!tm) {
        set_error("allocation failed");
        return NULL;
    }

    tm->n_clauses = n_clauses;
    tm->threshold = threshold;
    tm->s = s;
    tm->state_bits = state_bits;
    tm->boost_tpf = boost_tpf ? 1 : 0;
    tm->n_thresholds_per_feature = n_thresholds_per_feature;
    tm->task = task;
    tm->seed = seed;

    return tm;
}

EXPORT void wl_tm_free(WlTM *tm) {
    if (!tm) return;
    free(tm->thresholds);
    free(tm->threshold_counts);
    free(tm->ta_state);
    free(tm->class_labels);
    free(tm);
}

/* Initialize TA state: all automata at middle state (just below include threshold) */
static void init_ta_state(unsigned int *ta_state, int total_uints, int state_bits) {
    /* Set to state (1 << (state_bits-1)) - 1, which is just below the include threshold.
     * The include threshold is the MSB being set, so state_bits-1 is the boundary.
     * Initialize to half: set bits 0..state_bits-2 to 1 (state = 2^(state_bits-1) - 1) */
    memset(ta_state, 0, total_uints * sizeof(unsigned int));

    /* We need to set all automata to the initial state.
     * For bit-packed representation, each bit plane b stores bit b of all 32 automata.
     * State = (1 << (state_bits-1)) - 1: bits 0..state_bits-2 are all 1, MSB is 0.
     */
    /* Actually, TMU initializes via Python. We initialize all bits 0..state_bits-2
     * to 0xFFFFFFFF (all 32 automata in that chunk have that bit set). */

    /* Number of la_chunks is embedded in the layout. We iterate over all state bit planes. */
    /* The ta_state layout: for each clause, la_chunks groups of state_bits uints.
     * We don't know la_chunks here, so just set bits 0..state_bits-2 in every group. */
    /* Simpler: just set every state_bits-th word pattern. */

    /* Actually the simplest correct init: set all uints to 0xFFFFFFFF for bits 0..state_bits-2,
     * and 0 for bit state_bits-1 (the MSB / action bit). */
    /* Layout per chunk: [bit0, bit1, ..., bit_{state_bits-1}]
     * So for state = (1<<(sb-1))-1: bits 0..sb-2 = all 1s, bit sb-1 = 0 */

    /* But the total_uints might not be a nice multiple. Let's just be safe. */
    /* We don't have la_chunks info here, so we iterate the flat array. */
    /* Each group of state_bits uints represents one chunk of 32 automata.
     * Word i within a group = bit plane i. We want planes 0..sb-2 = 0xFFFFFFFF, plane sb-1 = 0. */
    int group_size = state_bits;
    int n_groups = total_uints / group_size;
    for (int g = 0; g < n_groups; g++) {
        for (int b = 0; b < state_bits - 1; b++) {
            ta_state[g * group_size + b] = 0xFFFFFFFF;
        }
        /* MSB (action bit) stays 0 -- all excluded initially */
        ta_state[g * group_size + state_bits - 1] = 0;
    }
}

EXPORT int wl_tm_fit(
    WlTM *tm,
    const double *X, int nrow, int ncol,
    const double *y, int epochs
) {
    if (!tm) { set_error("null model"); return -1; }
    if (nrow < 1 || ncol < 1) { set_error("invalid dimensions"); return -1; }
    if (epochs < 1) { set_error("epochs must be >= 1"); return -1; }

    pcg32_seed(tm->seed);
    tm->n_features = ncol;

    /* ---- Step 1: Binarization (compute thresholds from training data) ---- */
    int max_t = tm->n_thresholds_per_feature;
    tm->thresholds = (double *)calloc(ncol * max_t, sizeof(double));
    tm->threshold_counts = (int *)calloc(ncol, sizeof(int));
    if (!tm->thresholds || !tm->threshold_counts) {
        set_error("allocation failed for thresholds");
        return -1;
    }

    /* Extract each column and compute thresholds */
    double *col_buf = (double *)malloc(nrow * sizeof(double));
    if (!col_buf) { set_error("allocation failed"); return -1; }

    int total_binary = 0;
    for (int f = 0; f < ncol; f++) {
        for (int i = 0; i < nrow; i++) col_buf[i] = X[i * ncol + f];
        int cnt = compute_thresholds_for_feature(
            col_buf, nrow, max_t, &tm->thresholds[f * max_t]
        );
        tm->threshold_counts[f] = cnt;
        total_binary += cnt;
    }
    free(col_buf);

    /* Handle case where no thresholds were generated (e.g., all-constant data) */
    if (total_binary == 0) {
        set_error("no binary features generated (all features constant?)");
        return -1;
    }

    tm->n_binary = total_binary;
    tm->n_literals = 2 * total_binary;
    tm->la_chunks = (tm->n_literals - 1) / 32 + 1;

    /* ---- Step 2: Determine classes / regression params ---- */
    if (tm->task == 0) {
        /* Classification: find unique labels */
        int *labels_sorted = (int *)malloc(nrow * sizeof(int));
        if (!labels_sorted) { set_error("allocation failed"); return -1; }
        for (int i = 0; i < nrow; i++) labels_sorted[i] = (int)y[i];
        qsort(labels_sorted, nrow, sizeof(int), cmp_int);

        int n_classes = 1;
        for (int i = 1; i < nrow; i++) {
            if (labels_sorted[i] != labels_sorted[i - 1]) n_classes++;
        }

        tm->class_labels = (int *)malloc(n_classes * sizeof(int));
        if (!tm->class_labels) { free(labels_sorted); set_error("allocation failed"); return -1; }
        tm->class_labels[0] = labels_sorted[0];
        int ci = 1;
        for (int i = 1; i < nrow; i++) {
            if (labels_sorted[i] != labels_sorted[i - 1]) {
                tm->class_labels[ci++] = labels_sorted[i];
            }
        }
        tm->n_classes = n_classes;
        free(labels_sorted);
    } else {
        /* Regression: scale to [0, T] */
        tm->n_classes = 1;
        tm->y_min = y[0];
        tm->y_max = y[0];
        for (int i = 1; i < nrow; i++) {
            if (y[i] < tm->y_min) tm->y_min = y[i];
            if (y[i] > tm->y_max) tm->y_max = y[i];
        }
        if (tm->y_min == tm->y_max) {
            tm->y_max = tm->y_min + 1.0; /* avoid div by zero */
        }
    }

    /* ---- Step 3: Allocate TA state ---- */
    int clause_state_size = tm->n_clauses * tm->la_chunks * tm->state_bits;
    int total_ta = tm->n_classes * clause_state_size;
    tm->ta_state = (unsigned int *)malloc(total_ta * sizeof(unsigned int));
    if (!tm->ta_state) { set_error("allocation failed for ta_state"); return -1; }
    init_ta_state(tm->ta_state, total_ta, tm->state_bits);

    /* ---- Step 4: Pre-binarize all training examples ---- */
    unsigned int *Xi_all = (unsigned int *)calloc(nrow * tm->la_chunks, sizeof(unsigned int));
    if (!Xi_all) { set_error("allocation failed for Xi_all"); return -1; }

    for (int i = 0; i < nrow; i++) {
        binarize_row(
            &X[i * ncol], ncol,
            tm->thresholds, tm->threshold_counts,
            max_t,
            &Xi_all[i * tm->la_chunks], tm->n_literals, tm->la_chunks
        );
    }

    /* ---- Step 5: Training loop ---- */
    int n_clauses = tm->n_clauses;
    int half_clauses = n_clauses / 2;
    int n_literals = tm->n_literals;
    int state_bits = tm->state_bits;
    int T = tm->threshold;
    float s_f = (float)tm->s;

    /* Allocate working buffers */
    unsigned int *clause_output = (unsigned int *)malloc(n_clauses * sizeof(unsigned int));
    unsigned int *feedback_to_ta = (unsigned int *)malloc(tm->la_chunks * sizeof(unsigned int));
    unsigned int *output_one_patches = (unsigned int *)malloc(sizeof(unsigned int)); /* 1 patch */
    unsigned int *clause_active_pos = (unsigned int *)malloc(n_clauses * sizeof(unsigned int));
    unsigned int *clause_active_neg = (unsigned int *)malloc(n_clauses * sizeof(unsigned int));
    unsigned int *literal_active = (unsigned int *)malloc(tm->la_chunks * sizeof(unsigned int));

    if (!clause_output || !feedback_to_ta || !output_one_patches ||
        !clause_active_pos || !clause_active_neg || !literal_active) {
        free(Xi_all); free(clause_output); free(feedback_to_ta);
        free(output_one_patches); free(clause_active_pos); free(clause_active_neg);
        free(literal_active);
        set_error("allocation failed for training buffers");
        return -1;
    }

    /* Polarity masks: first N/2 positive, last N/2 negative (TMU convention) */
    for (int j = 0; j < n_clauses; j++) {
        clause_active_pos[j] = (j < half_clauses) ? 1 : 0;
        clause_active_neg[j] = (j < half_clauses) ? 0 : 1;
    }

    /* All literals active (no dropout) */
    memset(literal_active, 0xFF, tm->la_chunks * sizeof(unsigned int));

    int number_of_patches = 1; /* no convolution, single patch */

    for (int epoch = 0; epoch < epochs; epoch++) {
        /* Shuffle order using Fisher-Yates */
        int *order = (int *)malloc(nrow * sizeof(int));
        if (!order) { set_error("allocation failed"); goto cleanup_fail; }
        for (int i = 0; i < nrow; i++) order[i] = i;
        for (int i = nrow - 1; i > 0; i--) {
            int j2 = fast_rand() % (i + 1);
            int tmp = order[i]; order[i] = order[j2]; order[j2] = tmp;
        }

        for (int idx = 0; idx < nrow; idx++) {
            int i = order[idx];
            unsigned int *Xi = &Xi_all[i * tm->la_chunks];

            if (tm->task == 0) {
                /* ---- Classification training ---- */
                int target_label = (int)y[i];

                /* Find class index for target */
                int target_class = -1;
                for (int c = 0; c < tm->n_classes; c++) {
                    if (tm->class_labels[c] == target_label) {
                        target_class = c;
                        break;
                    }
                }
                if (target_class < 0) continue;

                /* Pick a random negative class */
                int neg_class = target_class;
                if (tm->n_classes > 1) {
                    while (neg_class == target_class) {
                        neg_class = fast_rand() % tm->n_classes;
                    }
                }

                /* === Target class === */
                unsigned int *ta_tgt = &tm->ta_state[target_class * clause_state_size];
                cb_calculate_clause_outputs_update(
                    ta_tgt, n_clauses, n_literals, state_bits,
                    number_of_patches, clause_output, literal_active, Xi
                );

                /* Vote sum: positive clauses (first half) +1, negative (second half) -1 */
                int vote_sum = 0;
                for (int j = 0; j < half_clauses; j++) {
                    if (clause_output[j]) vote_sum += 1;
                }
                for (int j = half_clauses; j < n_clauses; j++) {
                    if (clause_output[j]) vote_sum -= 1;
                }
                vote_sum = clamp_int(vote_sum, -T, T);

                /* Target: update_p = (T - vote_sum) / (2T) */
                float up_tgt = ((float)(T - vote_sum)) / (2.0f * T);

                /* Type I on positive clauses (strengthen what matches) */
                cb_type_i_feedback(
                    ta_tgt, feedback_to_ta, output_one_patches,
                    n_clauses, n_literals, state_bits, number_of_patches,
                    up_tgt, s_f, tm->boost_tpf, 0,
                    n_literals,
                    clause_active_pos, literal_active, Xi
                );

                /* Type II on negative clauses (weaken false positives) */
                cb_type_ii_feedback(
                    ta_tgt, output_one_patches,
                    n_clauses, n_literals, state_bits, number_of_patches,
                    up_tgt, clause_active_neg, literal_active, Xi
                );

                /* === Negative class === */
                unsigned int *ta_neg = &tm->ta_state[neg_class * clause_state_size];
                cb_calculate_clause_outputs_update(
                    ta_neg, n_clauses, n_literals, state_bits,
                    number_of_patches, clause_output, literal_active, Xi
                );

                int neg_vote_sum = 0;
                for (int j = 0; j < half_clauses; j++) {
                    if (clause_output[j]) neg_vote_sum += 1;
                }
                for (int j = half_clauses; j < n_clauses; j++) {
                    if (clause_output[j]) neg_vote_sum -= 1;
                }
                neg_vote_sum = clamp_int(neg_vote_sum, -T, T);

                /* Negative: update_p = (T + neg_vote_sum) / (2T) */
                float up_neg = ((float)(T + neg_vote_sum)) / (2.0f * T);

                /* Type I on negative clauses (reinforce what discriminates) */
                cb_type_i_feedback(
                    ta_neg, feedback_to_ta, output_one_patches,
                    n_clauses, n_literals, state_bits, number_of_patches,
                    up_neg, s_f, tm->boost_tpf, 0,
                    n_literals,
                    clause_active_neg, literal_active, Xi
                );

                /* Type II on positive clauses (punish wrong activations) */
                cb_type_ii_feedback(
                    ta_neg, output_one_patches,
                    n_clauses, n_literals, state_bits, number_of_patches,
                    up_neg, clause_active_pos, literal_active, Xi
                );

            } else {
                /* ---- Regression training ---- */
                double y_scaled = (y[i] - tm->y_min) / (tm->y_max - tm->y_min) * T;
                int y_target = clamp_int((int)round(y_scaled), 0, T);

                unsigned int *ta_reg = &tm->ta_state[0]; /* single class */
                cb_calculate_clause_outputs_update(
                    ta_reg, n_clauses, n_literals, state_bits,
                    number_of_patches, clause_output, literal_active, Xi
                );

                /* Vote sum: positive clauses - negative clauses, clamped to [0, T] */
                int vote_sum = 0;
                for (int j = 0; j < half_clauses; j++) {
                    if (clause_output[j]) vote_sum += 1;
                }
                for (int j = half_clauses; j < n_clauses; j++) {
                    if (clause_output[j]) vote_sum -= 1;
                }
                vote_sum = clamp_int(vote_sum, 0, T);

                /* Update probabilities based on error */
                float p_up = (float)clamp_int(y_target - vote_sum, 0, T) / (float)T;
                float p_down = (float)clamp_int(vote_sum - y_target, 0, T) / (float)T;

                if (p_up > 0) {
                    /* Need more votes: Type I on positive, Type II on negative */
                    cb_type_i_feedback(
                        ta_reg, feedback_to_ta, output_one_patches,
                        n_clauses, n_literals, state_bits, number_of_patches,
                        p_up, s_f, tm->boost_tpf, 0, n_literals,
                        clause_active_pos, literal_active, Xi
                    );
                    cb_type_ii_feedback(
                        ta_reg, output_one_patches,
                        n_clauses, n_literals, state_bits, number_of_patches,
                        p_up, clause_active_neg, literal_active, Xi
                    );
                }
                if (p_down > 0) {
                    /* Too many votes: Type I on negative, Type II on positive */
                    cb_type_i_feedback(
                        ta_reg, feedback_to_ta, output_one_patches,
                        n_clauses, n_literals, state_bits, number_of_patches,
                        p_down, s_f, tm->boost_tpf, 0, n_literals,
                        clause_active_neg, literal_active, Xi
                    );
                    cb_type_ii_feedback(
                        ta_reg, output_one_patches,
                        n_clauses, n_literals, state_bits, number_of_patches,
                        p_down, clause_active_pos, literal_active, Xi
                    );
                }
            }
        }
        free(order);
    }

    /* Cleanup training buffers */
    free(Xi_all);
    free(clause_output);
    free(feedback_to_ta);
    free(output_one_patches);
    free(clause_active_pos);
    free(clause_active_neg);
    free(literal_active);
    return 0;

cleanup_fail:
    free(Xi_all);
    free(clause_output);
    free(feedback_to_ta);
    free(output_one_patches);
    free(clause_active_pos);
    free(clause_active_neg);
    free(literal_active);
    return -1;
}

/* ---- Predict: compute class votes for one sample ---- */

static void predict_one(
    const WlTM *tm, const unsigned int *Xi,
    int *votes /* [n_classes] */
) {
    int n_clauses = tm->n_clauses;
    int half_clauses = n_clauses / 2;
    int n_literals = tm->n_literals;
    int state_bits = tm->state_bits;
    int number_of_patches = 1;
    int clause_state_size = n_clauses * tm->la_chunks * state_bits;

    unsigned int *clause_output = (unsigned int *)malloc(n_clauses * sizeof(unsigned int));
    if (!clause_output) return;

    for (int c = 0; c < tm->n_classes; c++) {
        unsigned int *ta_c = &tm->ta_state[c * clause_state_size];
        cb_calculate_clause_outputs_predict(
            ta_c, n_clauses, n_literals, state_bits,
            number_of_patches, clause_output, (unsigned int *)Xi
        );

        /* Contiguous polarity: first half positive, second half negative */
        int vote_sum = 0;
        for (int j = 0; j < half_clauses; j++) {
            if (clause_output[j]) vote_sum += 1;
        }
        for (int j = half_clauses; j < n_clauses; j++) {
            if (clause_output[j]) vote_sum -= 1;
        }
        votes[c] = vote_sum;
    }
    free(clause_output);
}

EXPORT int wl_tm_predict(
    const WlTM *tm,
    const double *X, int nrow, int ncol,
    int *out
) {
    if (!tm || !tm->ta_state) { set_error("model not fitted"); return -1; }
    if (ncol != tm->n_features) { set_error("feature count mismatch"); return -1; }

    unsigned int *Xi = (unsigned int *)calloc(tm->la_chunks, sizeof(unsigned int));
    int *votes = (int *)calloc(tm->n_classes, sizeof(int));
    if (!Xi || !votes) { free(Xi); free(votes); set_error("allocation failed"); return -1; }

    for (int i = 0; i < nrow; i++) {
        binarize_row(
            &X[i * ncol], ncol,
            tm->thresholds, tm->threshold_counts,
            tm->n_thresholds_per_feature,
            Xi, tm->n_literals, tm->la_chunks
        );

        memset(votes, 0, tm->n_classes * sizeof(int));
        predict_one(tm, Xi, votes);

        if (tm->task == 0) {
            /* Classification: argmax */
            int best_class = 0;
            int best_vote = votes[0];
            for (int c = 1; c < tm->n_classes; c++) {
                if (votes[c] > best_vote) {
                    best_vote = votes[c];
                    best_class = c;
                }
            }
            out[i] = tm->class_labels[best_class];
        } else {
            /* Regression: map vote sum back to original scale */
            int vote_sum = clamp_int(votes[0], 0, tm->threshold);
            double y_pred = tm->y_min + (double)vote_sum / (double)tm->threshold * (tm->y_max - tm->y_min);
            /* Store as raw double bits in int pairs: out[2*i], out[2*i+1] */
            /* Actually, let's use a separate predict function for regression */
            out[i] = vote_sum; /* raw vote (scaled back in JS) */
        }
    }

    free(Xi);
    free(votes);
    return 0;
}

EXPORT int wl_tm_predict_votes(
    const WlTM *tm,
    const double *X, int nrow, int ncol,
    int *out /* [nrow * n_classes] */
) {
    if (!tm || !tm->ta_state) { set_error("model not fitted"); return -1; }
    if (ncol != tm->n_features) { set_error("feature count mismatch"); return -1; }

    unsigned int *Xi = (unsigned int *)calloc(tm->la_chunks, sizeof(unsigned int));
    if (!Xi) { set_error("allocation failed"); return -1; }

    for (int i = 0; i < nrow; i++) {
        binarize_row(
            &X[i * ncol], ncol,
            tm->thresholds, tm->threshold_counts,
            tm->n_thresholds_per_feature,
            Xi, tm->n_literals, tm->la_chunks
        );

        predict_one(tm, Xi, &out[i * tm->n_classes]);
    }

    free(Xi);
    return 0;
}

/* ---- Serialization (TM01 format) ---- */

static const char TM01_MAGIC[4] = {'T', 'M', '0', '1'};
#define TM01_HEADER_SIZE 64

static void write_u32(uint8_t *buf, uint32_t val) {
    buf[0] = val & 0xFF;
    buf[1] = (val >> 8) & 0xFF;
    buf[2] = (val >> 16) & 0xFF;
    buf[3] = (val >> 24) & 0xFF;
}

static uint32_t read_u32(const uint8_t *buf) {
    return (uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24);
}

static void write_f64(uint8_t *buf, double val) {
    memcpy(buf, &val, 8);
}

static double read_f64(const uint8_t *buf) {
    double val;
    memcpy(&val, buf, 8);
    return val;
}

EXPORT int wl_tm_save(const WlTM *tm, uint8_t **out_buf, int *out_len) {
    if (!tm || !tm->ta_state) { set_error("model not fitted"); return -1; }

    /* Calculate total size */
    int clause_state_size = tm->n_classes * tm->n_clauses * tm->la_chunks * tm->state_bits;

    /* Count actual thresholds stored */
    int total_thresholds = 0;
    for (int f = 0; f < tm->n_features; f++) {
        total_thresholds += tm->threshold_counts[f];
    }

    size_t size = TM01_HEADER_SIZE;
    size += tm->n_features * sizeof(int32_t);          /* threshold_counts */
    size += total_thresholds * sizeof(double);          /* thresholds (packed) */
    size += clause_state_size * sizeof(uint32_t);       /* ta_state */
    if (tm->task == 0) {
        size += tm->n_classes * sizeof(int32_t);        /* class_labels */
    }

    uint8_t *buf = (uint8_t *)malloc(size);
    if (!buf) { set_error("allocation failed"); return -1; }
    memset(buf, 0, size);

    /* Header */
    memcpy(buf, TM01_MAGIC, 4);
    write_u32(buf + 4, 1); /* version */
    write_u32(buf + 8, tm->n_clauses);
    write_u32(buf + 12, tm->n_features);
    write_u32(buf + 16, tm->n_classes);
    write_u32(buf + 20, tm->n_binary);
    write_u32(buf + 24, tm->state_bits);
    write_u32(buf + 28, tm->threshold);
    write_f64(buf + 32, tm->s);
    buf[40] = (uint8_t)tm->task;
    buf[41] = (uint8_t)tm->boost_tpf;
    buf[42] = (uint8_t)(tm->n_thresholds_per_feature & 0xFF);
    buf[43] = (uint8_t)((tm->n_thresholds_per_feature >> 8) & 0xFF);
    /* reserved: 44..47 */
    write_f64(buf + 48, tm->y_min);
    write_f64(buf + 56, tm->y_max);

    /* Body */
    size_t pos = TM01_HEADER_SIZE;

    /* threshold_counts */
    for (int f = 0; f < tm->n_features; f++) {
        write_u32(buf + pos, (uint32_t)tm->threshold_counts[f]);
        pos += 4;
    }

    /* thresholds (packed: only threshold_counts[f] per feature) */
    for (int f = 0; f < tm->n_features; f++) {
        for (int t = 0; t < tm->threshold_counts[f]; t++) {
            write_f64(buf + pos, tm->thresholds[f * tm->n_thresholds_per_feature + t]);
            pos += 8;
        }
    }

    /* ta_state */
    for (int i = 0; i < clause_state_size; i++) {
        write_u32(buf + pos, tm->ta_state[i]);
        pos += 4;
    }

    /* class_labels (classification only) */
    if (tm->task == 0) {
        for (int c = 0; c < tm->n_classes; c++) {
            write_u32(buf + pos, (uint32_t)tm->class_labels[c]);
            pos += 4;
        }
    }

    *out_buf = buf;
    *out_len = (int)pos;
    return 0;
}

EXPORT WlTM *wl_tm_load(const uint8_t *buf, int len) {
    if (len < TM01_HEADER_SIZE) { set_error("buffer too small"); return NULL; }
    if (memcmp(buf, TM01_MAGIC, 4) != 0) { set_error("invalid TM01 magic"); return NULL; }

    uint32_t version = read_u32(buf + 4);
    if (version != 1) { set_error("unsupported TM01 version"); return NULL; }

    WlTM *tm = (WlTM *)calloc(1, sizeof(WlTM));
    if (!tm) { set_error("allocation failed"); return NULL; }

    tm->n_clauses = (int)read_u32(buf + 8);
    tm->n_features = (int)read_u32(buf + 12);
    tm->n_classes = (int)read_u32(buf + 16);
    tm->n_binary = (int)read_u32(buf + 20);
    tm->state_bits = (int)read_u32(buf + 24);
    tm->threshold = (int)read_u32(buf + 28);
    tm->s = read_f64(buf + 32);
    tm->task = buf[40];
    tm->boost_tpf = buf[41];
    tm->n_thresholds_per_feature = (int)buf[42] | ((int)buf[43] << 8);
    tm->y_min = read_f64(buf + 48);
    tm->y_max = read_f64(buf + 56);

    tm->n_literals = 2 * tm->n_binary;
    tm->la_chunks = (tm->n_literals - 1) / 32 + 1;

    size_t pos = TM01_HEADER_SIZE;

    /* threshold_counts */
    tm->threshold_counts = (int *)malloc(tm->n_features * sizeof(int));
    if (!tm->threshold_counts) { wl_tm_free(tm); set_error("allocation failed"); return NULL; }
    for (int f = 0; f < tm->n_features; f++) {
        if (pos + 4 > (size_t)len) { wl_tm_free(tm); set_error("truncated buffer"); return NULL; }
        tm->threshold_counts[f] = (int)read_u32(buf + pos);
        pos += 4;
    }

    /* thresholds (packed) */
    int max_t = tm->n_thresholds_per_feature;
    tm->thresholds = (double *)calloc(tm->n_features * max_t, sizeof(double));
    if (!tm->thresholds) { wl_tm_free(tm); set_error("allocation failed"); return NULL; }
    for (int f = 0; f < tm->n_features; f++) {
        for (int t = 0; t < tm->threshold_counts[f]; t++) {
            if (pos + 8 > (size_t)len) { wl_tm_free(tm); set_error("truncated buffer"); return NULL; }
            tm->thresholds[f * max_t + t] = read_f64(buf + pos);
            pos += 8;
        }
    }

    /* ta_state */
    int clause_state_size = tm->n_classes * tm->n_clauses * tm->la_chunks * tm->state_bits;
    tm->ta_state = (unsigned int *)malloc(clause_state_size * sizeof(unsigned int));
    if (!tm->ta_state) { wl_tm_free(tm); set_error("allocation failed"); return NULL; }
    for (int i = 0; i < clause_state_size; i++) {
        if (pos + 4 > (size_t)len) { wl_tm_free(tm); set_error("truncated buffer"); return NULL; }
        tm->ta_state[i] = read_u32(buf + pos);
        pos += 4;
    }

    /* class_labels (classification only) */
    if (tm->task == 0) {
        tm->class_labels = (int *)malloc(tm->n_classes * sizeof(int));
        if (!tm->class_labels) { wl_tm_free(tm); set_error("allocation failed"); return NULL; }
        for (int c = 0; c < tm->n_classes; c++) {
            if (pos + 4 > (size_t)len) { wl_tm_free(tm); set_error("truncated buffer"); return NULL; }
            tm->class_labels[c] = (int)read_u32(buf + pos);
            pos += 4;
        }
    }

    return tm;
}

EXPORT void wl_tm_free_buffer(void *ptr) {
    free(ptr);
}

/* ---- Getters ---- */

EXPORT int wl_tm_get_n_features(const WlTM *tm) { return tm ? tm->n_features : 0; }
EXPORT int wl_tm_get_n_classes(const WlTM *tm) { return tm ? tm->n_classes : 0; }
EXPORT int wl_tm_get_n_clauses(const WlTM *tm) { return tm ? tm->n_clauses : 0; }
EXPORT int wl_tm_get_task(const WlTM *tm) { return tm ? tm->task : 0; }
EXPORT int wl_tm_get_n_binary(const WlTM *tm) { return tm ? tm->n_binary : 0; }
EXPORT int wl_tm_get_threshold(const WlTM *tm) { return tm ? tm->threshold : 0; }
EXPORT double wl_tm_get_y_min(const WlTM *tm) { return tm ? tm->y_min : 0.0; }
EXPORT double wl_tm_get_y_max(const WlTM *tm) { return tm ? tm->y_max : 0.0; }
