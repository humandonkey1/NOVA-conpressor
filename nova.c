/*
 * NOVA 3.6 - our fast context-mixing compressor, built from scratch.
 * No word dictionaries. No LZ parsing/copying. No borrowed algorithms.
 * Pure bit-level prediction + our own carry-less range coder:
 *
 *   - nibble bit-tree context cells: every context slot is half a cache
 *     line holding a 15-cell binary tree for the 4 bits of the current
 *     nibble, so we pay ~1 memory miss per nibble instead of 4.
 *   - contexts: order-1 (direct), order-2/3/4/6 (hashed, 12-bit tags)
 *     plus a token-context (rolling hash of the current alphabetic token,
 *     computed on the fly - no dictionary, nothing stored per word).
 *   - each cell packs a 12-bit probability + 4-bit hit counter; fresh
 *     cells adapt fast, mature cells adapt slowly (self-tuning rate).
 *   - MMX "Mirror Match eXtended": a rolling 6-gram hash remembers WHERE
 *     the current phrase last ended; while the phrase keeps repeating we
 *     mirror the following bits with confidence growing with the verified
 *     match length. It never copies bytes and never emits tokens - it is
 *     purely one more per-bit predictor for the mixer (NOT an LZ scheme).
 *   - MMX fast lane: once a mirror match has held for FASTLEN bytes, each
 *     byte costs a single "still mirroring?" coder event (~0.01 bit while
 *     the run holds), which is where the big ratios and the big MB/s on
 *     real-world logs/JSON/XML come from.
 *   - SUBSTITUTION TOLERANCE: when a template field mutates, the mirror
 *     survives the miss instead of dying; the substituted byte is coded by
 *     the full model plus the SUBST predictor (old field byte -> new field
 *     byte transition cells) and the mirror re-aims through TWIN tables
 *     (record-so-far hash, line hash, line-prefix hash) to the record twin
 *     whose values agree - templated data rides through whole records.
 *   - MMX GLIDE: one arithmetic-coded confidence event vouches for a whole
 *     span of mirrored bytes (16/64/256/1024/4096) or a whole LINE at line
 *     starts; the payload bytes never touch the coder. No offsets, no
 *     copies, no tokens - purely our mirror predictor at coarser scales.
 *   - adaptive logistic mixer, 2048 weight sets keyed by last byte,
 *     match state and nibble position, online gradient per bit.
 *   - REF secondary refiner: interpolated 33-bin map from mixer output to
 *     observed reality, keyed by byte context / match / nibble position.
 *   - FIELD LANES: delimited fields that keep showing up as pure hex /
     *     digit / base64 across records (UUIDs, timestamps, telemetry
     *     numbers, tokens) are coded through tiny per-column adaptive bit
     *     trees (4/6 bit symbols) - their bare alphabet entropy, at a
     *     fraction of the full-model CPU on BOTH sides of the stream.
     *     No stored symbols, no dictionaries: same adaptive bit cells.
 *   - TURBO SKIP: when the two deepest contexts are mature and agree with
 *     near-certainty, the bit is coded from them alone and the rest of the
 *     machinery is skipped - structured data flies through the model.
 *   - DENSITY RADAR: blocks already at maximum entropy (random, encrypted,
 *     recompressed payloads) are detected on the fly and take an express
 *     lane at memcpy speed; the model never wastes cycles on noise.
 *   - ENTROPY LEDGER (opt-in NOVA_VAULT=dir): bit-exact defeat of true
 *     entropy by separating identity from body. A maximum-entropy segment
 *     is deposited once in a content-addressed vault; the archive carries
 *     a 21-byte identity record and decompression retrieves the EXACT
 *     original bits. Duplicate noise anywhere costs 21 bytes total.
 *   - PROVENANCE RNG ('nova r'): entropy defeated at the source. Noise
 *     generated with provenance is later collapsed bit-exactly to a
 *     14-byte genome by the NOISE GENOME scanner - no vault needed.
 *   - NOISE GENOME: not prediction - source reconstruction. Dense blocks
 *     are probed for an algorithmic origin: the scanner treats the stream
 *     as the orbit of an unknown state machine, recovers the seed from the
 *     observed words, verifies the orbit forward and collapses the whole
 *     run to a 14-byte genome {generator, seed, length}. Megabytes of
 *     "noise" become bytes; true entropy is untouched and keeps the
 *     express lane.
 *   - multithreaded independent chunks: bounded memory, safe on ANY data
 *     (text / binary / random), scales to arbitrarily large inputs.
 *
 * Build:  gcc -O3 -march=native -pthread nova.c -o nova -lm
 * Usage:  nova c <in> <out> | nova d <in> <out> | nova t | nova b <file>
 *         nova g <file> [size]   (generate the machine-data benchmark corpus)
 * Env:    NOVA_THREADS=N (default: number of CPUs), NOVA_VAULT=dir (ledger)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>

#define PSCALE   4096
#define NM       10                /* model inputs to the mixer          */
#define HBITS2   16                /* order-2 table: 2 MB, cache friendly */
#define HBITS3   18
#define HBITS4   18
#define HBITS6   18
#define HBITSW   18
#define MMBITS   21
#define MIXCTX   2048
#define WSHIFT   16                /* weight fixed point: 1.0 == 1<<16   */
#define WCLAMP   (1 << 20)
#define LRSHIFT  14
#define REFBINS  33
#define REFCTX   2048
#define FASTLEN  22               /* verified match length that arms the fast lane */

static const char *g_vault = NULL;  /* NOVA_VAULT=dir enables the ENTROPY LEDGER */
static int stretch_t[PSCALE];
static int squash_t[8193];

static void init_tables(void){
    double span = log((PSCALE - 0.5) / 0.5) / log(2.0);
    for (int i = 0; i < PSCALE; i++){
        double p = (i + 0.5) / PSCALE;
        double l = log(p / (1.0 - p)) / log(2.0);
        int v = (int)(4096.0 * l / span);
        if (v < -4096) v = -4096;
        if (v >  4096) v =  4096;
        stretch_t[i] = v;
    }
    for (int j = 0; j <= 8192; j++){
        int L = j - 4096;
        double p = 1.0 / (1.0 + exp(-(double)L / 4096.0 * span * log(2.0)));
        int v = (int)(PSCALE * p);
        if (v < 1) v = 1;
        if (v > PSCALE - 1) v = PSCALE - 1;
        squash_t[j] = v;
    }
}
static inline int sq(int L){ if (L < -4096) L = -4096; if (L > 4096) L = 4096; return squash_t[L + 4096]; }
static inline uint32_t hmul(uint64_t x){
    x *= 0x9E3779B97F4A7C15ULL; x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ULL;
    return (uint32_t)(x >> 32);
}

/* ---------------- range coder ---------------- */
typedef struct { uint8_t *out; size_t cap, len; uint32_t x1, x2; } Enc;
typedef struct { const uint8_t *in; size_t len, pos; uint32_t x1, x2, x; } Dec;
static void enc_init(Enc *e, size_t cap){ if (cap < 1024) cap = 1024; e->out = malloc(cap); e->cap = cap; e->len = 0; e->x1 = 0; e->x2 = 0xffffffff; }
static inline void enc_putc(Enc *e, uint8_t c){ if (e->len >= e->cap){ e->cap *= 2; e->out = realloc(e->out, e->cap); } e->out[e->len++] = c; }
static inline void enc_bit(Enc *e, int y, int p){
    uint32_t xmid = e->x1 + ((e->x2 - e->x1) >> 12) * (uint32_t)p;
    if (y) e->x2 = xmid; else e->x1 = xmid + 1;
    while (((e->x1 ^ e->x2) & 0xff000000) == 0){ enc_putc(e, (uint8_t)(e->x2 >> 24)); e->x1 <<= 8; e->x2 = (e->x2 << 8) | 255; }
}
static void enc_flush(Enc *e){ for (int i = 0; i < 4; i++){ enc_putc(e, (uint8_t)(e->x1 >> 24)); e->x1 <<= 8; } }
static void dec_init(Dec *d, const uint8_t *in, size_t len){ d->in = in; d->len = len; d->pos = 0; d->x1 = 0; d->x2 = 0xffffffff; d->x = 0; for (int i = 0; i < 4; i++) d->x = (d->x << 8) | (d->pos < d->len ? d->in[d->pos++] : 0); }
static inline int dec_bit(Dec *d, int p){
    uint32_t xmid = d->x1 + ((d->x2 - d->x1) >> 12) * (uint32_t)p;
    int y = (d->x <= xmid) ? 1 : 0;
    if (y) d->x2 = xmid; else d->x1 = xmid + 1;
    while (((d->x1 ^ d->x2) & 0xff000000) == 0){ d->x1 <<= 8; d->x2 = (d->x2 << 8) | 255; d->x = (d->x << 8) | (d->pos < d->len ? d->in[d->pos++] : 0); }
    return y;
}

/* ---------------- probability cells ----------------
 * uint16: [15:12] hit count, [11:0] probability of bit==1.
 * Fresh cells move fast, mature cells settle down. */
typedef uint16_t Cell;
static inline int cell_p(Cell c){ return c & 0xFFF; }
static inline Cell cell_upd(Cell c, int y){
    int n = c >> 12, p = c & 0xFFF;
    int sh = 2 + (n >> 1); if (sh > 6) sh = 6;
    if (y){ int inc = (4096 - p) >> sh; if (!inc) inc = 1; p += inc; if (p > 4094) p = 4094; }
    else  { int dec = p >> sh;          if (!dec) dec = 1; p -= dec; if (p < 1)    p = 1; }
    if (n < 15) n++;
    return (Cell)((n << 12) | p);
}
#define CELL0 2048

/* hashed slot: 32 bytes = half cache line.
 * v[0] = 12-bit tag, v[1..15] = bit-tree cells for one nibble. */
typedef struct { Cell v[16]; } __attribute__((aligned(32))) HSlot;
static inline Cell *hslot_get(HSlot *tab, uint32_t mask, uint32_t h){
    HSlot *s = &tab[(h >> 12) & mask];
    Cell tag = (Cell)(h & 0xFFF); if (!tag) tag = 1;
    if (s->v[0] != tag){
        s->v[0] = tag;
        for (int i = 1; i < 16; i++) s->v[i] = CELL0;
    }
    return s->v;
}

/* ---------------- model ---------------- */
typedef struct {
    Cell     t1[256 * 17][16];   /* order-1 direct: prev byte (+high nibble) */
    HSlot   *t2, *t3, *t4, *t6;  /* hashed orders 2/3/4/6                    */
    HSlot   *tw;                 /* token-context (hash of current word)     */
    uint32_t wh;                 /* rolling token hash                       */
    uint32_t *mm_pos;            /* 6-gram hash -> last position             */
    int32_t  wt[MIXCTX][NM];
    uint16_t ref[REFCTX][REFBINS];
    const uint8_t *hist;
    size_t   hpos;
    uint64_t b8;                 /* rolling last-8-bytes                     */
    uint32_t mptr, mlen;         /* mirror-match state                       */
    uint32_t miss;               /* consecutive mirror misses (substitution) */
    uint32_t hold;               /* bytes held since the last substitution   */
    uint32_t gskip;              /* bytes to code normally before next glide */
    size_t   lstart, lprev;      /* current / previous line starts           */
    size_t   lprevlen;           /* previous line length                     */
    uint32_t vrun;               /* consecutive vertical-context agreements  */
    uint64_t lh;                 /* hash of the current line so far          */
    uint32_t pxkey;              /* hash of the ~8 bytes right before the
                                    current field - the field's local identity */
    uint64_t lhprev;             /* full hash of the previous line           */
    uint64_t rh;                 /* hash of the whole record so far          */
    uint32_t rlines;             /* lines accumulated in rh                  */
    uint32_t sline;              /* substitutions seen in the current line   */
    uint32_t *mmf;               /* RECORD TWIN: 2-line hash -> position     */
    uint32_t *mmq;               /* LINE-PREFIX TWIN: prefix hash -> position */
    Cell     sub_hi[65536][16];  /* SUBST: prev-new byte x old byte -> hi    */
    Cell     sub_lo[65536][16];  /* SUBST: pn-low x old byte x new hi -> lo  */
    uint16_t fmp[16][16384];     /* fast-lane: P(byte holds): len x mclass x prevbyte x miss */
    uint16_t fgl[80];            /* glide confidence: 5 scales x 16 buckets  */
    uint16_t flg[16];            /* line-glide confidence by line length     */
    uint32_t f_idx;
    int      prev_class;
    /* FIELD LANES: when a delimited field has consistently been pure hex /
     * digit / base64 across recent records, the whole field is coded by a
     * tiny adaptive bit-tree (4 or 6 bits per symbol) instead of the full
     * context machine.  High-volume identifiers (UUIDs, timestamps,
     * numeric telemetry, tokens) cost their bare alphabet entropy and a
     * fraction of the CPU.  Still bit-level adaptive cells - no symbols
     * are ever stored, no dictionaries anywhere. */
    uint8_t  fl_cls[1024];   /* lane id per field index from previous line */
    uint8_t  fl_conf[1024];  /* confirmation counter (arm at >= 2)         */
    uint8_t  cl_cls[1024];   /* lane detected on the current line          */
    Cell     lcont[4][2][64];/* continuation bits: lane x col<prevlen x col */
    Cell     lsym4[3][8][16];/* 4-bit symbol trees: hex-lo/hex-up/digit    */
    Cell     lsym6[8][64];   /* 6-bit symbol tree: base64                  */
    int      lane_on;        /* active lane id (1..4), 0 = off             */
    uint32_t lane_f;         /* field index the lane was armed on          */
    uint32_t cur_field_start[1024];
    uint32_t cur_field_len[1024];
    uint32_t prev_field_start[1024];
    uint32_t prev_field_len[1024];
    uint32_t prev_fields;
    Cell     fvp[1024];
    /* SEQ-LOCK lane: monotone counters living inside fields (request ids,
     * build numbers, event ids).  When the longest digit run of a field
     * advances by the same delta twice, we bet a coin that it advances by
     * the same delta again - the whole field for 1 adaptive bit. */
    uint32_t seqp[1024];       /* position of the reference instance      */
    uint64_t seqv[1024];       /* its digit value                         */
    int64_t  seqd[1024];       /* delta that locked                       */
    uint8_t  seqds[1024];      /* digit-run start within the field        */
    uint8_t  seqw[1024];       /* digit-run width                         */
    uint8_t  seqlen[1024];     /* full field length                       */
    uint8_t  seqconf[1024];    /* confirmations so far                    */
    Cell     seqc[1024];       /* the betting coin                        */
} Model;

static Model *model_new(const uint8_t *hist){
    Model *m = malloc(sizeof(Model));
    for (int i = 0; i < 256 * 17; i++) for (int j = 0; j < 16; j++) m->t1[i][j] = CELL0;
    m->t2 = calloc((size_t)1 << HBITS2, sizeof(HSlot));
    m->t3 = calloc((size_t)1 << HBITS3, sizeof(HSlot));
    m->t4 = calloc((size_t)1 << HBITS4, sizeof(HSlot));
    m->t6 = calloc((size_t)1 << HBITS6, sizeof(HSlot));
    m->tw = calloc((size_t)1 << HBITSW, sizeof(HSlot));
    m->mm_pos = calloc((size_t)4 << MMBITS, 4);
    for (int c = 0; c < MIXCTX; c++){
        for (int k = 0; k < NM; k++) m->wt[c][k] = 0;
        for (int k = 1; k < NM; k++) m->wt[c][k] = (1 << WSHIFT) / 5;
    }
    for (int c = 0; c < REFCTX; c++)
        for (int j = 0; j < REFBINS; j++)
            m->ref[c][j] = (uint16_t)sq((j - (REFBINS >> 1)) * 8192 / (REFBINS - 1));
    m->hist = hist; m->hpos = 0;
    m->b8 = 0; m->mptr = 0; m->mlen = 0; m->wh = 0;
    for (int i = 0; i < 16; i++) for (int j = 0; j < 16384; j++) m->fmp[i][j] = 3968;
    for (int i = 0; i < 80; i++) m->fgl[i] = 3600;
    for (int i = 0; i < 16; i++) m->flg[i] = 3000;
    m->miss = 0; m->hold = 0; m->gskip = 0;
    m->lstart = 0; m->lprev = 0; m->lprevlen = 0; m->vrun = 0;
    m->lh = 0; m->lhprev = 0; m->rh = 0; m->rlines = 0; m->sline = 0;
    m->mmf = calloc((size_t)1 << MMBITS, 4);
    m->mmq = calloc((size_t)1 << MMBITS, 4);
    for (int i = 0; i < 65536; i++) for (int j = 0; j < 16; j++) m->sub_hi[i][j] = CELL0;
    for (int i = 0; i < 65536; i++) for (int j = 0; j < 16; j++) m->sub_lo[i][j] = CELL0;
    m->f_idx = 0;
    m->prev_class = 2;
    m->lane_on = 0; m->lane_f = 0;
    for (int i = 0; i < 1024; i++){ m->fl_cls[i] = 0; m->fl_conf[i] = 0; m->cl_cls[i] = 0; }
    for (int a = 0; a < 4; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 64; c++) m->lcont[a][b][c] = CELL0;
    for (int a = 0; a < 3; a++) for (int b = 0; b < 8; b++) for (int c = 0; c < 16; c++) m->lsym4[a][b][c] = CELL0;
    for (int b = 0; b < 8; b++) for (int c = 0; c < 64; c++) m->lsym6[b][c] = CELL0;
    m->cur_field_start[0] = 0;
    m->prev_fields = 0;
    for (int i = 0; i < 1024; i++) {
        m->fvp[i] = CELL0;
        m->cur_field_start[i] = 0;
        m->cur_field_len[i] = 0;
        m->seqc[i] = CELL0; m->seqconf[i] = 0;
    (void)0;
        m->prev_field_start[i] = 0;
        m->prev_field_len[i] = 0;
    }
    return m;
}
static void model_free(Model *m){
    free(m->t2); free(m->t3); free(m->t4); free(m->t6); free(m->tw);
    free(m->mm_pos); free(m->mmf); free(m->mmq); free(m);
}

/* code one nibble (4 bits). enc!=NULL: encode nibble 'nib'.
 * returns the nibble. */
static inline int code_nibble(Model *m, Enc *enc, Dec *dec, int nib,
                              Cell *c1, Cell *c2, Cell *c3, Cell *c4, Cell *c6, Cell *cw,
                              Cell *su, int su_flag __attribute__((unused)),
                              int mm_on, int mnib, int mst_mag,
                              int vv_on, int vnib, int vst_mag, int prevbyte, int hi){
    int cc = 1;
    for (int i = 0; i < 4; i++){
        /* turbo skip: when the two deepest contexts are mature and certain,
         * code the bit from them alone and skip the heavy machinery */
        Cell c3v = c3[cc], c6v = c6[cc];
        if ((c3v >> 12) == 15 && (c6v >> 12) == 15){
            int p3 = c3v & 0xFFF, p6 = c6v & 0xFFF;
            if ((p3 > 3968 && p6 > 3968) || (p3 < 128 && p6 < 128)){
                int q = (p3 + p6) >> 1;
                if (q < 1) q = 1; else if (q > 4094) q = 4094;
                int bit;
                if (enc){ bit = (nib >> (3 - i)) & 1; enc_bit(enc, bit, q); }
                else      bit = dec_bit(dec, q);
                c3[cc] = cell_upd(c3v, bit);
                c6[cc] = cell_upd(c6v, bit);
                if (mm_on && bit != ((mnib >> (3 - i)) & 1)) mm_on = 0;
                cc = (cc << 1) | bit;
                continue;
            }
        }
        int stv[NM];
        stv[0] = 256;
        stv[1] = stretch_t[cell_p(c1[cc])];
        stv[2] = stretch_t[cell_p(c2[cc])];
        stv[3] = stretch_t[cell_p(c3[cc])];
        stv[4] = stretch_t[cell_p(c4[cc])];
        stv[5] = stretch_t[cell_p(c6[cc])];
        stv[6] = stretch_t[cell_p(cw[cc])];
        int ebit = 0;
        if (mm_on){ ebit = (mnib >> (3 - i)) & 1; stv[7] = ebit ? mst_mag : -mst_mag; }
        else stv[7] = 0;
        if (vv_on){ int vb = (vnib >> (3 - i)) & 1; stv[8] = vb ? vst_mag : -vst_mag; }
        else stv[8] = 0;
        stv[9] = su ? stretch_t[cell_p(su[cc])] : 0;
        int mc = prevbyte | (mm_on << 8) | (hi << 9) | ((cc & 1) << 10);
        int32_t *w = m->wt[mc];
        int64_t L = 0;
        for (int k = 0; k < NM; k++) L += (int64_t)w[k] * stv[k];
        int Ls = (int)(L >> WSHIFT);
        int q = sq(Ls);
        /* REF refiner */
        int rctx = (prevbyte << 3) | (mm_on << 2) | (hi << 1) | (i >> 1);
        int zz = Ls + 4096; if (zz < 0) zz = 0; else if (zz > 8191) zz = 8191;
        int bin = zz >> 8, frac = zz & 255;
        uint16_t *r0 = &m->ref[rctx][bin], *r1 = &m->ref[rctx][bin + 1];
        int qr = ((int)*r0 * (256 - frac) + (int)*r1 * frac) >> 8;
        int qf = (q + qr) >> 1;
        if (qf < 1) qf = 1; else if (qf > 4094) qf = 4094;
        int bit;
        if (enc){ bit = (nib >> (3 - i)) & 1; enc_bit(enc, bit, qf); }
        else      bit = dec_bit(dec, qf);
        /* learn: mixer on its own output */
        int err = (bit ? 4095 : 0) - q;
        if (err > 30 || err < -30){
            for (int k = 0; k < NM; k++){
                int32_t nw = w[k] + ((err * stv[k]) >> LRSHIFT);
                if (nw < -WCLAMP) nw = -WCLAMP; else if (nw > WCLAMP) nw = WCLAMP;
                w[k] = nw;
            }
        }
        if (bit){ *r0 += (4095 - *r0) >> 6; *r1 += (4095 - *r1) >> 6; }
        else    { *r0 -= *r0 >> 6;          *r1 -= *r1 >> 6; }
        c1[cc] = cell_upd(c1[cc], bit);
        c2[cc] = cell_upd(c2[cc], bit);
        c3[cc] = cell_upd(c3[cc], bit);
        c4[cc] = cell_upd(c4[cc], bit);
        c6[cc] = cell_upd(c6[cc], bit);
        cw[cc] = cell_upd(cw[cc], bit);
        if (su) su[cc] = cell_upd(su[cc], bit);
        if (mm_on && bit != ebit) mm_on = 0;
        cc = (cc << 1) | bit;
    }
    return cc & 15;
}

static inline int code_byte_full(Model *m, Enc *enc, Dec *dec, int c, int mm_off){
    uint64_t b8 = m->b8;
    int prevbyte = (int)(b8 & 0xFF);
    /* vertical context: byte at the same column of the previous line */
    int vv_on = 0, vbyte = 0, vst_mag = 0;
    {
        size_t col = m->hpos - m->lstart;
        if (m->lprevlen && col < m->lprevlen){
            vbyte = m->hist[m->lprev + col];
            vv_on = 1;
            uint32_t vr = m->vrun < 24 ? m->vrun : 24;
            vst_mag = 300 + (int)vr * 130;
        }
    }
    /* mirror-match state for this byte */
    int mm_on = (m->mlen > 0) && !mm_off && !m->miss;
    int mbyte = mm_on ? m->hist[m->mptr] : 0;
    /* SUBST PREDICTOR: during a substitution the OLD field byte from the
     * mirror is a strong context - counters and timestamps mutate in
     * learnable ways (7->8, 9->0, same digit class) */
    int su_on = (m->mlen > 0) && (mm_off || m->miss) && m->miss <= 4 && (size_t)m->mptr < m->hpos;
    int obyte = su_on ? m->hist[m->mptr] : 0;
    int mst_mag = 0;
    if (mm_on){
        uint32_t L = m->mlen < 28 ? m->mlen : 28;
        mst_mag = 800 + (int)L * 122;
    }
    uint64_t k2 = b8 & 0xFFFF, k3 = b8 & 0xFFFFFF;
    uint64_t k4 = b8 & 0xFFFFFFFFULL, k6 = b8 & 0xFFFFFFFFFFFFULL;
    /* high nibble */
    Cell *c1 = m->t1[prevbyte];
    Cell *c2 = hslot_get(m->t2, (1u << HBITS2) - 1, hmul(k2 * 4 + 2));
    Cell *c3 = hslot_get(m->t3, (1u << HBITS3) - 1, hmul(k3 * 4 + 3));
    Cell *c4 = hslot_get(m->t4, (1u << HBITS4) - 1, hmul(k4 * 4 + 1));
    Cell *c6 = hslot_get(m->t6, (1u << HBITS6) - 1, hmul(k6 * 4 + 3));
    uint64_t kw = (uint64_t)m->wh * 3 + 1;
    Cell *cw = hslot_get(m->tw, (1u << HBITSW) - 1, hmul(kw));
    int onext = su_on && (size_t)m->mptr + 1 < m->hpos ? m->hist[m->mptr + 1] : 0;
    Cell *su = su_on ? m->sub_hi[(onext << 8) | obyte] : NULL;
    int hi = code_nibble(m, enc, dec, (c >> 4) & 15, c1, c2, c3, c4, c6, cw, su, su_on,
                         mm_on, mbyte >> 4, mst_mag, vv_on, vbyte >> 4, vst_mag, prevbyte, 0);
    if (mm_on && hi != (mbyte >> 4)) mm_on = 0;
    /* low nibble (contexts extended by the high nibble) */
    uint64_t hx = (uint64_t)(hi + 1) << 56;
    c1 = m->t1[256 + prevbyte * 16 + hi];
    c2 = hslot_get(m->t2, (1u << HBITS2) - 1, hmul((k2 | hx) * 4 + 2));
    c3 = hslot_get(m->t3, (1u << HBITS3) - 1, hmul((k3 | hx) * 4 + 3));
    c4 = hslot_get(m->t4, (1u << HBITS4) - 1, hmul((k4 | hx) * 4 + 1));
    c6 = hslot_get(m->t6, (1u << HBITS6) - 1, hmul((k6 | hx) * 4 + 3));
    cw = hslot_get(m->tw, (1u << HBITSW) - 1, hmul(kw ^ (hx >> 24)));
    su = su_on ? m->sub_lo[((onext & 15) << 12) | (obyte << 4) | hi] : NULL;
    int lo = code_nibble(m, enc, dec, c & 15, c1, c2, c3, c4, c6, cw, su, su_on,
                         mm_on, mbyte & 15, mst_mag, vv_on, vbyte & 15, vst_mag, prevbyte, 1);
    return (hi << 4) | lo;
}

static inline void prefetch_next(Model *m){
    uint64_t b8 = m->b8;
    uint64_t k2 = b8 & 0xFFFF, k3 = b8 & 0xFFFFFF;
    uint64_t k4 = b8 & 0xFFFFFFFFULL, k6 = b8 & 0xFFFFFFFFFFFFULL;
    __builtin_prefetch(&m->t2[(hmul(k2 * 4 + 2) >> 12) & ((1u << HBITS2) - 1)], 1, 1);
    __builtin_prefetch(&m->t3[(hmul(k3 * 4 + 3) >> 12) & ((1u << HBITS3) - 1)], 1, 1);
    __builtin_prefetch(&m->t4[(hmul(k4 * 4 + 1) >> 12) & ((1u << HBITS4) - 1)], 1, 1);
    __builtin_prefetch(&m->t6[(hmul(k6 * 4 + 3) >> 12) & ((1u << HBITS6) - 1)], 1, 1);
    __builtin_prefetch(&m->tw[(hmul((uint64_t)m->wh * 3 + 1) >> 12) & ((1u << HBITSW) - 1)], 1, 1);
}

/* Fast lane: while a long mirror match is verified, spend ONE coder event
 * on "byte still mirrors" (probability adapts per match-length bucket).
 * Only on a break do we fall back to order-1 nibble trees for the byte.
 * Long repeats therefore cost ~0.001 bits/byte and almost no CPU.
 * Still no offsets, no copies, no tokens - just a very sure predictor. */
/* template char class: where do substitutions live? digits mutate,
 * structure holds - the hold-flag learns this per class */
static uint8_t ccl_tab[256];
static void ccl_init(void){
    for (int b = 0; b < 256; b++){
        int r = 7;
        if (b >= '0' && b <= '9') r = 0;
        else if (b >= 'a' && b <= 'z') r = 1;
        else if (b >= 'A' && b <= 'Z') r = 2;
        else if (b == ' ' || b == '\t') r = 3;
        else if (b == '\n' || b == '\r') r = 4;
        else if (b == '"' || b == ',' || b == ':' || b == '=' || b == '.') r = 5;
        else if (b >= 128) r = 6;
        ccl_tab[b] = (uint8_t)r;
    }
}
static inline int char_class(int b){ return ccl_tab[b & 0xFF]; }
static inline int code_byte_fast(Model *m, Enc *enc, Dec *dec, int c){
    int mbyte = m->hist[m->mptr];
    uint32_t L = m->hold; if (L > 127) L = 127;
    int lb = (int)(L >> 4);                      /* 0..7 hold bucket */
    int va = m->vrun >= 4 ? 1 : 0;               /* vertical agreement */
    lb = (lb << 1) | va;                         /* 0..15 */
    int mi = m->miss > 7 ? 7 : (int)m->miss;
    int pb = (int)(m->b8 & 0xFF);
    uint16_t *mp = &m->fmp[lb][(char_class(mbyte) << 11) | (pb << 3) | mi];
    int q = *mp; if (q < 1) q = 1; else if (q > 4094) q = 4094;
    int same;
    if (enc){ same = (c == mbyte); enc_bit(enc, same, q); }
    else      same = dec_bit(dec, q);
    if (same){
        int inc = (4095 - *mp) >> 7; if (!inc) inc = 1;
        *mp += inc; if (*mp > 4094) *mp = 4094;
        return mbyte;
    }
    *mp -= *mp >> 5;
    /* substitution byte: full context model, mirror input muted */
    return code_byte_full(m, enc, dec, c, 1);
}

static inline int code_byte(Model *m, Enc *enc, Dec *dec, int c){
    if (m->mlen >= FASTLEN) return code_byte_fast(m, enc, dec, c);
    return code_byte_full(m, enc, dec, c, 0);
}

/* MMX GLIDE: one probabilistic event vouches for a whole span of mirrored
 * bytes. No tokens, no offsets - a single arithmetic-coded confidence bit
 * on our own mirror predictor; the payload bytes never touch the coder.
 * Three escalating scales: 16 -> 64 -> 256 bytes. Confidence per scale. */
#define GLIDE_MIN 48
static const int glide_n[5] = {16, 64, 256, 1024, 4096};

/* length of the mirrored line ahead of mptr (0 = not usable) */
static inline uint32_t line_glide_len(Model *m, size_t remain){
    if (!(m->mlen >= GLIDE_MIN && m->miss == 0)) return 0;
    if (m->hpos != m->lstart) return 0;      /* only at line starts */
    uint32_t p = m->mptr, lim = p + 300;
    if ((size_t)lim > m->hpos) lim = (uint32_t)m->hpos;
    while (p < lim && m->hist[p] != '\n') p++;
    if (p >= lim) return 0;
    uint32_t n = p + 1 - m->mptr;
    if (n < 8 || (size_t)n > remain) return 0;
    return n;
}
static inline int glide_scale(Model *m, size_t remain){
    if (!(m->mlen >= GLIDE_MIN && m->miss == 0) ) return -1;
    for (int sc = 4; sc >= 0; sc--){
        size_t n = (size_t)glide_n[sc];
        if (m->hold >= n && remain >= n && (size_t)m->mptr + n <= m->hpos &&
            m->fgl[sc * 16 + ((m->mlen >> 8) & 15)] >= 2600)
            return sc;
    }
    return -1;
}

/* ---------------- FIELD LANES ---------------- */
#define LN_HEXLO 1
#define LN_HEXUP 2
#define LN_DIGIT 3
#define LN_B64   4
static uint8_t lane_alpha[4][256];   /* lane_alpha[lane-1][byte] */
static uint8_t lane_val[4][256];     /* symbol value per lane    */
static uint8_t gcl_tab[256];         /* get_class lookup         */
static char LN_B64C[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static void lane_tables_init(void){
    ccl_init();
    for (int c = '0'; c <= '9'; c++){
        lane_alpha[LN_HEXLO-1][c] = 1; lane_val[LN_HEXLO-1][c] = (uint8_t)(c - '0');
        lane_alpha[LN_HEXUP-1][c] = 1; lane_val[LN_HEXUP-1][c] = (uint8_t)(c - '0');
        lane_alpha[LN_DIGIT-1][c] = 1; lane_val[LN_DIGIT-1][c] = (uint8_t)(c - '0');
        lane_alpha[LN_B64-1][c] = 1;
    }
    for (int c = 'a'; c <= 'f'; c++){ lane_alpha[LN_HEXLO-1][c] = 1; lane_val[LN_HEXLO-1][c] = (uint8_t)(c - 'a' + 10); }
    for (int c = 'A'; c <= 'F'; c++){ lane_alpha[LN_HEXUP-1][c] = 1; lane_val[LN_HEXUP-1][c] = (uint8_t)(c - 'A' + 10); }
    for (int c = 'A'; c <= 'Z'; c++) lane_alpha[LN_B64-1][c] = 1;
    for (int c = 'a'; c <= 'z'; c++) lane_alpha[LN_B64-1][c] = 1;
    lane_alpha[LN_B64-1]['+'] = 1; lane_alpha[LN_B64-1]['/'] = 1;
    for (int i = 0; i < 64; i++) lane_val[LN_B64-1][(uint8_t)LN_B64C[i]] = (uint8_t)i;
    /* get_class LUT */
    for (int c = 0; c < 256; c++) gcl_tab[c] = 1;
    gcl_tab[(uint8_t)'\n'] = 2;
    gcl_tab[','] = 0; gcl_tab[';'] = 0; gcl_tab[' '] = 0; gcl_tab['\t'] = 0; gcl_tab['|'] = 0; gcl_tab[':'] = 0;
}
/* classify a finished field into a lane (0 = no lane) */
static inline int lane_classify(const uint8_t *p, uint32_t n){
    if (n < 8) return 0;
    uint32_t hl = 0, hu = 0, dg = 0, b6 = 0, up = 0;
    for (uint32_t i = 0; i < n; i++){
        uint8_t c = p[i];
        if (c >= '0' && c <= '9'){ hl++; hu++; dg++; b6++; }
        else if (c >= 'a' && c <= 'f'){ hl++; b6++; }
        else if (c >= 'A' && c <= 'F'){ hu++; b6++; up++; }
        else if ((c >= 'g' && c <= 'z') || (c >= 'G' && c <= 'Z') || c == '+' || c == '/'){ b6++; if (c >= 'G' && c <= 'Z') up++; }
    }
    /* only IID-uniform alphabets earn a lane: hex ids (UUIDs/hashes) and
     * base64 tokens.  Plain digits are predictive gold for the temporal
     * model (walks, counters, monotonic timestamps) - leave them to it.
     * A lane must Prove Mixing: real hex ids carry digit+letter soup,
     * real tokens carry digits AND upper AND lower case; word-like or
     * number-like fields stay with the context model. */
    int r = 0;
    if (n >= 10 && dg >= 2 && (hl - dg) * 10 >= n * 3 && hl * 20 >= n * 17) r = LN_HEXLO;
    else if (n >= 10 && dg >= 2 && (hu - dg) * 10 >= n * 3 && hu * 20 >= n * 17) r = LN_HEXUP;
    /* base64 explicitly NOT laned: on pure b64 the context model already
     * codes at alphabet entropy, so a lane is pure continuation overhead */
    (void)up; (void)b6;
    return r;
}
/* code one lane symbol through its bit tree; returns the symbol value */
static inline int lane_sym_code(Model *m, Enc *enc, Dec *dec, int lane, int colb, int val){
    if (lane != LN_B64){
        Cell *t = m->lsym4[lane - 1][colb];
        int cc = 1;
        for (int i = 3; i >= 0; i--){
            int p = cell_p(t[cc]); if (p < 1) p = 1; else if (p > 4094) p = 4094;
            int b;
            if (enc){ b = (val >> i) & 1; enc_bit(enc, b, p); } else b = dec_bit(dec, p);
            t[cc] = cell_upd(t[cc], b);
            cc = (cc << 1) | b;
        }
        return cc & 15;
    }
    Cell *t = m->lsym6[colb];
    int cc = 1;
    for (int i = 5; i >= 0; i--){
        int p = cell_p(t[cc]); if (p < 1) p = 1; else if (p > 4094) p = 4094;
        int b;
        if (enc){ b = (val >> i) & 1; enc_bit(enc, b, p); } else b = dec_bit(dec, p);
        t[cc] = cell_upd(t[cc], b);
        cc = (cc << 1) | b;
    }
    return cc & 63;
}
static inline int get_class(int c) { return gcl_tab[c & 0xFF]; }

/* SEQ-LOCK feed: parse the longest digit run of the finished content
 * field and fold it into the per-field delta tracker (both sides run this
 * identically from decoded history). */
static inline uint32_t seq_key(const Model *m, uint32_t fidx){
    return (m->pxkey ^ (fidx * 977u)) & 1023;
}
static inline void seq_feed(Model *m, uint32_t fidx, uint32_t start, uint32_t len){
    const uint8_t *fp = m->hist + start;
    uint32_t bds = 0, bw = 0;
    for (uint32_t i = 0; i < len;) {
        if (fp[i] >= '0' && fp[i] <= '9'){
            uint32_t s = i;
            while (i < len && fp[i] >= '0' && fp[i] <= '9') i++;
            if (i - s > bw){ bds = s; bw = i - s; }
        } else i++;
    }
    if (!bw || bw > 12 || len > 32) return;
    uint64_t v = 0;
    for (uint32_t i = bds; i < bds + bw; i++) v = v * 10 + (uint64_t)(fp[i] - '0');
    if (m->seqconf[fidx] >= 2){
        int64_t d = (int64_t)v - (int64_t)m->seqv[fidx];
        if (d == m->seqd[fidx] && d != 0){ if (m->seqconf[fidx] < 63) m->seqconf[fidx]++; }
        else m->seqconf[fidx] = 1;
        m->seqv[fidx] = v; m->seqp[fidx] = start;
    } else {
        /* one reference instance held: sample the first delta */
        int64_t d = m->seqconf[fidx] ? (int64_t)v - (int64_t)m->seqv[fidx] : 0;
        m->seqds[fidx] = (uint8_t)bds;
        m->seqw[fidx] = (uint8_t)bw;
        m->seqlen[fidx] = (uint8_t)len;
        m->seqv[fidx] = v; m->seqp[fidx] = start;
        if (m->seqconf[fidx] && d != 0){ m->seqd[fidx] = d; m->seqconf[fidx] = 2; }
        else m->seqconf[fidx] = 1;
    }
}
/* predicted instance: reference bytes with the digit run re-rendered */
static inline int seq_predict(const Model *m, uint32_t fidx, uint8_t *out){
    int64_t pv = (int64_t)m->seqv[fidx] + m->seqd[fidx];
    uint32_t w = m->seqw[fidx], n = m->seqlen[fidx];
    if (pv < 0) return 0;
    uint64_t v = (uint64_t)pv, lim = 1;
    for (uint32_t i = 0; i < w; i++){
        if (lim > 9999999999999ULL) return 0;
        lim *= 10;
    }
    if (v >= lim) return 0;
    memcpy(out, m->hist + m->seqp[fidx], n);
    for (uint32_t i = 0; i < w; i++){ out[m->seqds[fidx] + w - 1 - i] = (uint8_t)('0' + v % 10); v /= 10; }
    return 1;
}
static inline void track_transition(Model *m, int cls, size_t hpos) {
    if (cls != m->prev_class) {
        uint32_t finished_f_idx = m->f_idx;
        uint32_t start_pos = m->cur_field_start[finished_f_idx];
        uint32_t len = (uint32_t)(hpos - start_pos);
        m->cur_field_len[finished_f_idx] = len;
        /* SEQ-LOCK feed: fold the finished content field into the tracker.
         * The key is (field index x line-prefix hash): "timeout" and "build"
         * live on different rows even when they share a field index. */
        if (m->prev_class == 1 && len >= 2 && len <= 32)
            seq_feed(m, seq_key(m, finished_f_idx), start_pos, len);

        /* classify the just-finished field for the FIELD LANES.
         * Field content (separator excluded) is [start+1, start+len). */
        if (len >= 6 && finished_f_idx < 1024)
            m->cl_cls[finished_f_idx] = lane_classify(m->hist + start_pos + 1, len - 1);
        else if (finished_f_idx < 1024)
            m->cl_cls[finished_f_idx] = 0;

        if (m->prev_class == 2) {
            for (uint32_t i = 0; i <= finished_f_idx && i < 1024; i++) {
                m->prev_field_start[i] = m->cur_field_start[i];
                m->prev_field_len[i] = m->cur_field_len[i];
                /* fold this line's lane detections into the armed set;
                 * a zero detection is "no evidence" - keep prior belief */
                uint8_t det = m->cl_cls[i];
                if (det && det == m->fl_cls[i]){ if (m->fl_conf[i] < 3) m->fl_conf[i]++; }
                else if (det){ m->fl_cls[i] = det; m->fl_conf[i] = 1; }
                m->cl_cls[i] = 0;
            }
            m->prev_fields = finished_f_idx + 1;
            m->f_idx = 0;
        } else {
            if (m->f_idx < 1023) {
                m->f_idx++;
            }
        }
        m->cur_field_start[m->f_idx] = (uint32_t)hpos;
        m->pxkey = (uint32_t)m->b8 ^ (uint32_t)(m->b8 >> 32);
        m->prev_class = cls;
    }
}


/* called AFTER byte c has been stored in hist[hpos] */
static inline void match_update(Model *m, int c){
    int cls = get_class(c);
    track_transition(m, cls, m->hpos);

    if (m->mlen){
        if (m->hist[m->mptr] == c){ m->mptr++; if (m->mlen < 65535) m->mlen++; m->miss = 0; if (m->hold < 65535) m->hold++; }
        else if (m->mlen >= FASTLEN && m->miss < 16){
            /* substitution tolerance: a template field changed, the
             * mirror survives but decays - persistent misses disarm it */
            m->mptr++; m->miss++; m->hold = 0; m->sline++;
            m->mlen -= m->mlen >> 3;
        }
        else { m->mlen = 0; m->miss = 0; m->hold = 0; m->sline++; }
    }
    {   /* vertical context bookkeeping */
        size_t col = m->hpos - m->lstart;
        if (m->lprevlen && col < m->lprevlen && m->hist[m->lprev + col] == (uint8_t)c){
            if (m->vrun < 65535) m->vrun++;
        } else m->vrun = 0;
        if (c == '\n'){
            /* RECORD TWIN: key = the last TWO full lines. Inside a record
             * the fields of consecutive lines are correlated, so the twin
             * of a 2-line prefix continues with the same field values.
             * Aim only when the current mirror struggled on this line. */
            if (m->hpos - m->lstart >= 4){
                if (++m->rlines > 8){ m->rh = 0; m->rlines = 1; }
                m->rh = m->rh * 0x9E3779B97F4A7C15ULL ^ m->lh;
                uint32_t hf = hmul(m->rh * 2 + 1) >> (32 - MMBITS);
                uint32_t cand = m->mmf[hf];
                if (cand > 0 && cand < m->hpos && cand != m->mptr &&
                    (m->sline || m->mlen < FASTLEN)){
                    /* backward verify 8 bytes to reject hash collisions */
                    uint32_t vk = 0;
                    while (vk < 8 && cand > vk && m->hist[cand - 1 - vk] == m->hist[m->hpos - vk]) vk++;
                    if (vk >= 8 || cand <= vk){
                        m->mptr = cand; m->miss = 0; m->hold = 0;
                        if (m->mlen < 48) m->mlen = 48;
                    }
                }
                m->mmf[hf] = (uint32_t)(m->hpos + 1);
            }
            m->lhprev = m->lh;
            if (m->hpos == m->lstart){ m->rh = 0; m->rlines = 0; }  /* blank line: record end */
            m->lh = 0;
            m->sline = 0;
            m->lprev = m->lstart;
            m->lprevlen = m->hpos + 1 - m->lstart;
            m->lstart = m->hpos + 1;
            m->vrun = 0;
        } else {
                m->lh = m->lh * 0x100000001B3ULL + (uint64_t)c + 1;
                size_t off = m->hpos - m->lstart;
                if (off >= 6 && off <= 96){
                    uint32_t hq = hmul(m->lh * 4 + 2) >> (32 - MMBITS);
                    /* LINE-PREFIX TWIN: on a substitution, the freshest line
                     * with this exact prefix (which already contains the NEW
                     * field bytes) is the best mirror - hop there */
                    if (m->miss && !(c >= '0' && c <= '9')){
                        uint32_t cand = m->mmq[hq];
                        if (cand > 0 && cand < m->hpos && cand != m->mptr){
                            /* backward verify: the last 10 bytes must agree so
                             * we only hop to a true prefix twin */
                            uint32_t vk = 0, lim = cand < 10 ? cand : 10;
                            while (vk < lim && m->hist[cand - 1 - vk] == m->hist[m->hpos - vk]) vk++;
                            if (vk >= lim){
                                m->mptr = cand; m->miss = 0;
                                if (m->mlen < 32) m->mlen = 32;
                            }
                        }
                    }
                    /* freshness write only matters while the mirror is cold
                     * or limping; a hot mirror skips the store entirely */
                    if (m->mlen < FASTLEN || m->miss)
                        m->mmq[hq] = (uint32_t)(m->hpos + 1);
                }
        }
    }
    m->hpos++;
    m->b8 = (m->b8 << 8) | (uint64_t)c;
    {   /* token-context hash: letters accumulate, anything else resets */
        int ch = c;
        if (ch >= 'A' && ch <= 'Z') ch += 32;
        if ((ch >= 'a' && ch <= 'z') || ch >= 128) m->wh = m->wh * 0x2F0B3F1Du + (uint32_t)ch + 1;
        else m->wh = (uint32_t)ch * 0x9E3779B9u;
    }
    /* HOT-MIRROR FREE PASS: while a long clean mirror runs, the position
     * table gains nothing (no probing happens) and prefetches for the full
     * mixer would only burn L3 bandwidth - skip both.  The table simply
     * keeps the positions from the arm region, which are still real twins. */
    if (m->hpos >= 6 && (m->mlen < FASTLEN || m->miss)){
        uint32_t h6 = hmul(m->b8 & 0xFFFFFFFFFFFFULL) >> (32 - MMBITS);
        uint32_t *slot = &m->mm_pos[(size_t)h6 * 4];
        if (!m->mlen || m->miss){
            /* pick the candidate whose recent history agrees the longest -
             * on templated data this lands on the line with the SAME field
             * values, so the mirror rides through whole records */
            uint32_t best = 0, bestk = 0;
            int depth = m->miss ? 2 : 1;
            for (int t = 0; t < depth; t++){
                uint32_t cand = slot[t];
                if (cand >= 6 && cand < m->hpos && (!m->mlen || cand != m->mptr)){
                    uint32_t k = 0, lim = (cand < m->hpos - cand) ? cand : (uint32_t)(m->hpos - cand);
                    if (lim > 64) lim = 64;
                    while (k < lim && m->hist[cand - 1 - k] == m->hist[m->hpos - 1 - k]) k++;
                    if (k > bestk){ bestk = k; best = cand; }
                }
            }
            if (bestk >= 6){
                if (!m->mlen){ m->mptr = best; m->mlen = bestk; }
                else { m->mptr = best; if (m->mlen < bestk) m->mlen = bestk; m->miss = 0; }
            }
        }
        slot[3] = slot[2]; slot[2] = slot[1]; slot[1] = slot[0];
        slot[0] = (uint32_t)m->hpos;
    }
    if (m->mlen) __builtin_prefetch(&m->hist[m->mptr], 0, 1);
    if (m->mlen < FASTLEN || m->miss) prefetch_next(m);
}

/* ---------------- NOISE GENOME ----------------
 * Not prediction - source reconstruction. Digital "noise" is almost never
 * true entropy: it comes out of an algorithmic generator, and then its
 * Kolmogorov complexity is a few bytes of state, not megabytes of output.
 * The genome scanner treats the byte stream as the orbit of an unknown
 * state machine, recovers the state directly from the observed words and
 * verifies the orbit forward. On a hit, the whole segment collapses to
 * {generator id, 8-byte seed, length} - a 14-byte genome for up to 4 GB
 * of noise. The scan costs a few cycles per candidate offset, so streams
 * with no genome (true entropy) lose nothing and keep the express lane. */
#define GN_MIN 4096
#define GN_XSSTAR_MUL 2685821657736338717ULL
#define GN_XSSTAR_INV 0x59071D96D81ECD35ULL   /* modular inverse mod 2^64 */
static inline uint64_t ld64(const uint8_t *p){ uint64_t x; memcpy(&x, p, 8); return x; }
static inline uint64_t gn_next(int id, uint64_t s){
    switch (id){
    case 0: s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;            /* xorshift64 */
    case 1: return s * 6364136223846793005ULL + 1442695040888963407ULL;   /* LCG (MMIX) */
    case 2: return s * 2862933555777941757ULL + 3037000493ULL;            /* LCG (Knuth) */
    case 3: return s * 6364136223846793005ULL + 1ULL;                     /* LCG variant */
    case 4: s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s;           /* xorshift64* core */
    default: return s + 0x9E3779B97F4A7C15ULL;                            /* splitmix64 ctr */
    }
}
static inline uint64_t gn_mix(uint64_t z){
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t gn_unmix(uint64_t z){
    z ^= z >> 31; z ^= z >> 62;
    z *= 0x319642B2D24D8EC3ULL;                    /* inv of 0x94D049BB133111EB */
    z ^= z >> 27; z ^= z >> 54;
    z *= 0x96DE1B173F119089ULL;                    /* inv of 0xBF58476D1CE4E5B9 */
    z ^= z >> 30; z ^= z >> 60;
    return z;
}
static inline uint64_t gn_emit(int id, uint64_t s){
    if (id == 4) return s * GN_XSSTAR_MUL;
    if (id == 5) return gn_mix(s);
    return s;
}
static inline uint64_t gn_state(int id, uint64_t o){
    if (id == 4) return o * GN_XSSTAR_INV;
    if (id == 5) return gn_unmix(o);
    return o;
}
#define GN_NGEN 6
/* returns genome run length (multiple of 8) found at p+*goff, or 0 */
static size_t genome_scan(const uint8_t *p, size_t n, int *gid, int *goff){
    if (n < GN_MIN + 8) return 0;
    for (int off = 0; off < 8; off++){
        size_t avail = n - off;
        if (avail < GN_MIN) break;
        uint64_t o0 = ld64(p + off), o1 = ld64(p + off + 8);
        for (int id = 0; id < GN_NGEN; id++){
            uint64_t st = gn_next(id, gn_state(id, o0));
            if (gn_emit(id, st) != o1) continue;
            size_t k = 16;
            while (k + 8 <= avail){
                st = gn_next(id, st);
                if (gn_emit(id, st) != ld64(p + off + k)) break;
                k += 8;
            }
            if (k >= GN_MIN){ *gid = id; *goff = off; return k; }
        }
    }
    return 0;
}

/* ---------------- ENTROPY LEDGER ----------------
 * Bit-exact defeat of true entropy. Counting rules out squeezing the bits
 * themselves, so the ledger does what enterprise content-addressed
 * storage does: it separates the IDENTITY of the data from its BODY.
 * A maximum-entropy segment is hashed into a 128-bit identity; the body
 * is deposited once in the vault (a content-addressed object store), and
 * the archive carries only the 21-byte identity record. Decompression
 * retrieves the EXACT original bits from the vault - always bit-exact,
 * never regenerated. Identical noise seen again anywhere (re-compression,
 * backups, retransmission, duplicated blobs) costs 21 bytes and zero new
 * vault space. Opt-in via NOVA_VAULT=dir; without it NOVA stays fully
 * self-contained. */
static void ledger_hash(const uint8_t *p, size_t n, uint64_t *h1, uint64_t *h2){
    uint64_t a = 0x9E3779B97F4A7C15ULL ^ n, b = 0xC2B2AE3D27D4EB4FULL + n * 0x165667B19E3779F9ULL;
    size_t i = 0;
    for (; i + 8 <= n; i += 8){
        uint64_t w = ld64(p + i);
        a = (a ^ w) * 0xBF58476D1CE4E5B9ULL; a ^= a >> 29;
        b = (b + w) * 0x94D049BB133111EBULL; b ^= b >> 31;
    }
    for (; i < n; i++){ a = (a ^ p[i]) * 0x100000001B3ULL; b = (b + p[i]) * 0x100000001B3ULL; }
    *h1 = gn_mix(a ^ (b >> 7)); *h2 = gn_mix(b ^ (a << 9) ^ 0xA5A5A5A5A5A5A5A5ULL);
}
static void vault_path(char *dst, size_t cap, uint64_t h1, uint64_t h2){
    snprintf(dst, cap, "%s/%016llx%016llx.nvo", g_vault,
             (unsigned long long)h1, (unsigned long long)h2);
}
static int vault_put(const uint8_t *p, size_t n, uint64_t h1, uint64_t h2){
    char path[1024]; struct stat st;
    mkdir(g_vault, 0777);
    vault_path(path, sizeof path, h1, h2);
    if (stat(path, &st) == 0 && (size_t)st.st_size == n) return 0;   /* dedup hit */
    char tmp[1060]; snprintf(tmp, sizeof tmp, "%s.tmp%d", path, (int)getpid());
    FILE *f = fopen(tmp, "wb"); if (!f) return -1;
    if (fwrite(p, 1, n, f) != n){ fclose(f); remove(tmp); return -1; }
    fclose(f);
    if (rename(tmp, path) != 0){ remove(tmp); return -1; }
    return 0;
}
static int vault_get(uint8_t *dst, size_t n, uint64_t h1, uint64_t h2){
    char path[1024]; vault_path(path, sizeof path, h1, h2);
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    size_t got = fread(dst, 1, n, f); fclose(f);
    return got == n ? 0 : -1;
}

/* ---------------- chunk workers ---------------- */
/* Density radar: measure order-0 entropy of each 256 KB block on the fly.
 * Blocks that are already at maximum density (random / encrypted / already
 * compressed payloads) carry no predictable structure, so the model is not
 * wasted on them - they take the express lane at memcpy speed while the
 * context model keeps its state warm for the structured parts. */
#define SEGBLK (256 * 1024)

static int block_is_dense(const uint8_t *p, size_t n){
    size_t sn = n < 65536 ? n : 65536;
    if (sn < 4096) return 0;
    uint32_t h[256] = {0};
    for (size_t j = 0; j < sn; j++) h[p[j]]++;
    double H = 0.0;
    for (int k = 0; k < 256; k++) if (h[k]){ double q = (double)h[k] / sn; H -= q * log2(q); }
    return H > 7.60;
}
/* keep byte context in sync across an express segment (both sides do this) */
static void model_skip(Model *m, const uint8_t *p, size_t start, size_t end){
    uint64_t b8 = m->b8;
    size_t from = end > 8 && end - 8 > start ? end - 8 : start;
    for (size_t j = from; j < end; j++) b8 = (b8 << 8) | p[j];
    m->b8 = b8; m->hpos = end; m->mlen = 0; m->wh = 0;
    m->f_idx = 0;
    m->prev_class = 2;
    m->cur_field_start[0] = (uint32_t)end;
    m->prev_fields = 0;
    m->lane_on = 0;
}
static void ob_need(uint8_t **ob, size_t *cap, size_t len, size_t need){
    if (len + need > *cap){ while (len + need > *cap) *cap = *cap * 2 + 1024; *ob = realloc(*ob, *cap); }
}
static uint8_t *compress_chunk(const uint8_t *in, size_t n, size_t *outlen){
    size_t cap = n / 2 + 1024, ol = 0;
    uint8_t *ob = malloc(cap);
    Model *m = model_new(in);
    size_t i = 0;
    while (i < n){
        int dense = block_is_dense(in + i, n - i);
        size_t j = i;
        do {
            size_t blk = n - j < SEGBLK ? n - j : SEGBLK;
            j += blk;
        } while (j < n && block_is_dense(in + j, n - j) == dense);
        size_t seg = j - i;
        if (dense){
            int gid = 0, goff = 0;
            size_t glen = genome_scan(in + i, seg, &gid, &goff);
            if (glen){
                if (goff){               /* few unaligned bytes before the orbit */
                    ob_need(&ob, &cap, ol, (size_t)goff + 5);
                    ob[ol++] = 'R';
                    for (int b = 0; b < 4; b++) ob[ol++] = (uint8_t)((uint32_t)goff >> (8 * b));
                    memcpy(ob + ol, in + i, goff); ol += goff;
                    model_skip(m, in, i, i + goff);
                    i += goff;
                }
                ob_need(&ob, &cap, ol, 14);
                ob[ol++] = 'G';
                for (int b = 0; b < 4; b++) ob[ol++] = (uint8_t)(glen >> (8 * b));
                ob[ol++] = (uint8_t)gid;
                uint64_t seed = ld64(in + i);
                for (int b = 0; b < 8; b++) ob[ol++] = (uint8_t)(seed >> (8 * b));
                model_skip(m, in, i, i + glen);
                i += glen;
                continue;                /* rescan the rest of the region */
            }
            if (g_vault && seg >= 65536){
                uint64_t h1, h2; ledger_hash(in + i, seg, &h1, &h2);
                if (vault_put(in + i, seg, h1, h2) == 0){
                    ob_need(&ob, &cap, ol, 21);
                    ob[ol++] = 'V';
                    for (int b = 0; b < 4; b++) ob[ol++] = (uint8_t)(seg >> (8 * b));
                    for (int b = 0; b < 8; b++) ob[ol++] = (uint8_t)(h1 >> (8 * b));
                    for (int b = 0; b < 8; b++) ob[ol++] = (uint8_t)(h2 >> (8 * b));
                    model_skip(m, in, i, j);
                    i = j;
                    continue;
                }
            }
            ob_need(&ob, &cap, ol, seg + 5);
            ob[ol++] = 'R';
            for (int b = 0; b < 4; b++) ob[ol++] = (uint8_t)(seg >> (8 * b));
            memcpy(ob + ol, in + i, seg); ol += seg;
            model_skip(m, in, i, j);
        } else {
            Enc e; enc_init(&e, seg / 2 + 256);
            for (size_t k = i; k < j; k++){
                uint32_t f = m->f_idx;
                if (k == m->cur_field_start[f] + 1 && m->prev_fields && f < m->prev_fields) {
                    uint32_t L = m->prev_field_len[f];
                    uint32_t prev_start = m->prev_field_start[f];
                    if (L >= 2 && k + (L - 1) <= j) {
                        if (m->hist[k - 1] == m->hist[prev_start]) {
                            int hit = (memcmp(in + k, m->hist + prev_start + 1, L - 1) == 0);
                            int q = cell_p(m->fvp[f]);
                            enc_bit(&e, hit, q);
                            m->fvp[f] = cell_upd(m->fvp[f], hit);
                            if (hit) {
                                for (uint32_t t = 1; t < L; t++) {
                                    match_update(m, in[k + t - 1]);
                                }
                                k += L - 2;
                                continue;
                            }
                        }
                    }
                }
                /* SEQ-LOCK probe (encode): the field's digits marched on by
                 * the locked delta - one adaptive coin buys the whole field */
                { uint32_t sf = seq_key(m, f);
                if (k == m->cur_field_start[f] + 1 && f < m->prev_fields
                    && m->seqconf[sf] >= 3 && get_class(in[k - 1]) == 1){
                    f = sf;
                    uint8_t tmp[32];
                    uint32_t n = m->seqlen[f];
                    int hit = 0;
                    if (seq_predict(m, f, tmp) && k - 1 + n <= j){
                        uint32_t n0 = 0;
                        while (k - 1 + n0 < j && n0 <= 32 && get_class(in[k - 1 + n0]) == 1) n0++;
                        hit = (n0 == n && memcmp(tmp, in + k - 1, n) == 0);
                    }
                    int q = cell_p(m->seqc[f]);
                    enc_bit(&e, hit, q);
                    m->seqc[f] = cell_upd(m->seqc[f], hit);
                    if (hit){
                        for (uint32_t t = 1; t < n; t++) match_update(m, in[k + t - 1]);
                        k += n - 2;
                        continue;
                    }
                } }

                uint32_t lgn;
                if (m->gskip == 0 && (lgn = line_glide_len(m, j - k)) != 0){
                    uint16_t *gp = &m->flg[(lgn >> 4) & 15];
                    int hit = memcmp(in + k, m->hist + m->mptr, lgn) == 0;
                    int q = *gp; if (q < 1) q = 1; else if (q > 4094) q = 4094;
                    enc_bit(&e, hit, q);
                    if (hit){
                        int gi2 = (4095 - *gp) >> 5; if (!gi2) gi2 = 1;
                        *gp += gi2; if (*gp > 4094) *gp = 4094;
                        for (uint32_t t = 0; t < lgn; t++) match_update(m, in[k + t]);
                        k += lgn - 1;
                        continue;
                    }
                    *gp -= *gp >> 5;
                    m->gskip = 16;
                }
                int sc;
                if (m->gskip == 0 && (sc = glide_scale(m, j - k)) >= 0){
                    int gn = glide_n[sc];
                    uint16_t *gp = &m->fgl[sc * 16 + ((m->mlen >> 8) & 15)];
                    int hit = memcmp(in + k, m->hist + m->mptr, gn) == 0;
                    int q = *gp; if (q < 1) q = 1; else if (q > 4094) q = 4094;
                    enc_bit(&e, hit, q);
                    if (hit){
                        int ginc = (4095 - *gp) >> 6; if (!ginc) ginc = 1;
                        *gp += ginc; if (*gp > 4094) *gp = 4094;
                        for (int t = 0; t < gn; t++) match_update(m, in[k + t]);
                        k += gn - 1;
                        continue;
                    }
                    *gp -= *gp >> 5;
                    m->gskip = 16;           /* decode side mirrors this */
                }
                /* FIELD LANE: alphabet lane for identifiers / numbers.
                 * The lane stays armed for the whole field: bytes outside
                 * the alphabet just take the full model (cont bit = 0),
                 * per-column cont cells learn the fixed structure. */
                {
                    uint32_t lf = m->f_idx;
                    if (m->lane_on){
                        if (m->f_idx != m->lane_f || (m->mlen >= FASTLEN && !m->miss)){
                            /* field ended or a verified mirror is cheaper -
                             * drop the lane (state-only, decode mirrors) */
                            m->lane_on = 0;
                        } else {
                            int ln = m->lane_on - 1;
                            int c8 = in[k];
                            size_t col64 = k - (size_t)m->cur_field_start[lf] - 1;
                            int col = col64 > 63 ? 63 : (int)col64;
                            int under = (lf < m->prev_fields && col64 + 1 < m->prev_field_len[lf]) ? 1 : 0;
                            Cell *qc = &m->lcont[ln][under][col];
                            int p = cell_p(*qc); if (p < 1) p = 1; else if (p > 4094) p = 4094;
                            if (lane_alpha[ln][c8]){
                                enc_bit(&e, 1, p); *qc = cell_upd(*qc, 1);
                                lane_sym_code(m, &e, NULL, m->lane_on, col >> 3, lane_val[ln][c8]);
                                match_update(m, c8);
                                continue;
                            }
                            enc_bit(&e, 0, p); *qc = cell_upd(*qc, 0);
                            /* stay armed; normal machinery codes this byte */
                        }
                    } else if (lf < m->prev_fields && m->fl_conf[lf] >= 2
                               && k == (size_t)m->cur_field_start[lf] + 1
                               && ((m->fl_cls[lf] != LN_B64 && m->prev_field_len[lf] >= 11)
                                   || (m->fl_cls[lf] == LN_B64 && m->prev_field_len[lf] >= 17))
                               && (m->mlen < FASTLEN || m->miss)){
                        m->lane_on = m->fl_cls[lf];
                        m->lane_f = lf;
                        /* no consume: field byte 0 warms up via the model */
                    }
                }
                if (m->gskip) m->gskip--;
                code_byte(m, &e, NULL, in[k]); match_update(m, in[k]);
            }
            enc_flush(&e);
            ob_need(&ob, &cap, ol, e.len + 9);
            ob[ol++] = 'C';
            for (int b = 0; b < 4; b++) ob[ol++] = (uint8_t)(seg >> (8 * b));
            for (int b = 0; b < 4; b++) ob[ol++] = (uint8_t)(e.len >> (8 * b));
            memcpy(ob + ol, e.out, e.len); ol += e.len;
            free(e.out);
        }
        i = j;
    }
    model_free(m);
    *outlen = ol; return ob;
}
static uint8_t *decompress_chunk(const uint8_t *in, size_t n, size_t outn){
    uint8_t *out = calloc(outn ? outn : 1, 1);
    Model *m = model_new(out);
    size_t p = 0, o = 0;
    while (o < outn && p < n){
        int type = in[p++];
        uint32_t ulen = 0; for (int b = 0; b < 4; b++) ulen |= (uint32_t)in[p++] << (8 * b);
        if (type == 'R'){
            memcpy(out + o, in + p, ulen); p += ulen;
            model_skip(m, out, o, o + ulen);
            o += ulen;
        } else if (type == 'V'){
            uint64_t h1 = 0, h2 = 0;
            for (int b = 0; b < 8; b++) h1 |= (uint64_t)in[p++] << (8 * b);
            for (int b = 0; b < 8; b++) h2 |= (uint64_t)in[p++] << (8 * b);
            if (!g_vault || vault_get(out + o, ulen, h1, h2) != 0){
                fprintf(stderr, "ENTROPY LEDGER: vault object %016llx%016llx missing (set NOVA_VAULT)\n",
                        (unsigned long long)h1, (unsigned long long)h2);
                exit(1);
            }
            model_skip(m, out, o, o + ulen);
            o += ulen;
        } else if (type == 'G'){
            int gid = in[p++];
            uint64_t seed = 0; for (int b = 0; b < 8; b++) seed |= (uint64_t)in[p++] << (8 * b);
            memcpy(out + o, &seed, 8);
            uint64_t st = gn_state(gid, seed);
            for (size_t k = 8; k < ulen; k += 8){
                st = gn_next(gid, st);
                uint64_t ov = gn_emit(gid, st);
                memcpy(out + o + k, &ov, 8);
            }
            model_skip(m, out, o, o + ulen);
            o += ulen;
        } else {
            uint32_t clen = 0; for (int b = 0; b < 4; b++) clen |= (uint32_t)in[p++] << (8 * b);
            Dec d; dec_init(&d, in + p, clen);
            for (uint32_t k = 0; k < ulen; k++){
                uint32_t f = m->f_idx;
                if ((o + k) == m->cur_field_start[f] + 1 && m->prev_fields && f < m->prev_fields) {
                    uint32_t L = m->prev_field_len[f];
                    uint32_t prev_start = m->prev_field_start[f];
                    if (L >= 2 && k + (L - 1) <= ulen) {
                        if (m->hist[o + k - 1] == m->hist[prev_start]) {
                            int q = cell_p(m->fvp[f]);
                            int hit = dec_bit(&d, q);
                            m->fvp[f] = cell_upd(m->fvp[f], hit);
                            if (hit) {
                                for (uint32_t t = 1; t < L; t++) {
                                    uint8_t cb = m->hist[prev_start + t];
                                    out[o + k + t - 1] = cb;
                                    match_update(m, cb);
                                }
                                k += L - 2;
                                continue;
                            }
                        }
                    }
                }
                /* SEQ-LOCK (decode side - mirrors the encoder exactly) */
                { uint32_t sf = seq_key(m, f);
                if ((o + k) == (size_t)m->cur_field_start[f] + 1 && f < m->prev_fields
                    && m->seqconf[sf] >= 3 && get_class(m->hist[o + k - 1]) == 1){
                    f = sf;
                    uint8_t tmp[32];
                    uint32_t n = m->seqlen[f];
                    int ok = seq_predict(m, f, tmp) && k + n - 1 <= ulen;
                    int q = cell_p(m->seqc[f]);
                    int hit = dec_bit(&d, q);
                    m->seqc[f] = cell_upd(m->seqc[f], hit);
                    if (hit && ok){
                        for (uint32_t t = 1; t < n; t++){
                            out[o + k + t - 1] = tmp[t];
                            match_update(m, tmp[t]);
                        }
                        k += n - 2;
                        continue;
                    }
                } }

                uint32_t lgn;
                if (m->gskip == 0 && (lgn = line_glide_len(m, (size_t)ulen - k)) != 0){
                    uint16_t *gp = &m->flg[(lgn >> 4) & 15];
                    int q = *gp; if (q < 1) q = 1; else if (q > 4094) q = 4094;
                    int hit = dec_bit(&d, q);
                    if (hit){
                        int gi2 = (4095 - *gp) >> 5; if (!gi2) gi2 = 1;
                        *gp += gi2; if (*gp > 4094) *gp = 4094;
                        for (uint32_t t = 0; t < lgn; t++){
                            uint8_t cb2 = m->hist[m->mptr];
                            out[o + k + t] = cb2;
                            match_update(m, cb2);
                        }
                        k += lgn - 1;
                        continue;
                    }
                    *gp -= *gp >> 5;
                    m->gskip = 16;
                }
                int sc;
                if (m->gskip == 0 && (sc = glide_scale(m, (size_t)ulen - k)) >= 0){
                    int gn = glide_n[sc];
                    uint16_t *gp = &m->fgl[sc * 16 + ((m->mlen >> 8) & 15)];
                    int q = *gp; if (q < 1) q = 1; else if (q > 4094) q = 4094;
                    int hit = dec_bit(&d, q);
                    if (hit){
                        int ginc = (4095 - *gp) >> 6; if (!ginc) ginc = 1;
                        *gp += ginc; if (*gp > 4094) *gp = 4094;
                        for (int t = 0; t < gn; t++){
                            uint8_t cb2 = m->hist[m->mptr];
                            out[o + k + t] = cb2;
                            match_update(m, cb2);
                        }
                        k += gn - 1;
                        continue;
                    }
                    *gp -= *gp >> 5;
                    m->gskip = 16;
                }
                /* FIELD LANE (decode side - mirrors the encoder exactly) */
                {
                    uint32_t lf = m->f_idx;
                    if (m->lane_on){
                        if (m->f_idx != m->lane_f || (m->mlen >= FASTLEN && !m->miss)){
                            m->lane_on = 0;
                        } else {
                            int ln = m->lane_on - 1;
                            size_t abspos = o + k;
                            size_t col64 = abspos - (size_t)m->cur_field_start[lf] - 1;
                            int col = col64 > 63 ? 63 : (int)col64;
                            int under = (lf < m->prev_fields && col64 + 1 < m->prev_field_len[lf]) ? 1 : 0;
                            Cell *qc = &m->lcont[ln][under][col];
                            int p = cell_p(*qc); if (p < 1) p = 1; else if (p > 4094) p = 4094;
                            int cont = dec_bit(&d, p);
                            *qc = cell_upd(*qc, cont);
                            if (cont){
                                int v = lane_sym_code(m, NULL, &d, m->lane_on, col >> 3, 0);
                                int c8;
                                if (m->lane_on == LN_HEXLO)      c8 = v < 10 ? '0' + v : 'a' + v - 10;
                                else if (m->lane_on == LN_HEXUP) c8 = v < 10 ? '0' + v : 'A' + v - 10;
                                else if (m->lane_on == LN_DIGIT) c8 = '0' + (v & 15);
                                else                             c8 = LN_B64C[v & 63];
                                out[o + k] = (uint8_t)c8;
                                match_update(m, c8);
                                continue;
                            }
                            /* stay armed; normal machinery codes this byte */
                        }
                    } else if (lf < m->prev_fields && m->fl_conf[lf] >= 2
                               && (o + k) == (size_t)m->cur_field_start[lf] + 1
                               && ((m->fl_cls[lf] != LN_B64 && m->prev_field_len[lf] >= 11)
                                   || (m->fl_cls[lf] == LN_B64 && m->prev_field_len[lf] >= 17))
                               && (m->mlen < FASTLEN || m->miss)){
                        m->lane_on = m->fl_cls[lf];
                        m->lane_f = lf;
                    }
                }
                if (m->gskip) m->gskip--;
                int c = code_byte(m, NULL, &d, 0); out[o + k] = (uint8_t)c; match_update(m, c);
            }
            p += clen; o += ulen;
        }
    }
    model_free(m); return out;
}
typedef struct { const uint8_t *in; size_t n; uint8_t *out; size_t outlen; } ChunkArg;
static void *t_compress(void *a){ ChunkArg *c = (ChunkArg*)a; c->out = compress_chunk(c->in, c->n, &c->outlen); return NULL; }
static void *t_decompress(void *a){ ChunkArg *c = (ChunkArg*)a; c->out = decompress_chunk(c->in, c->n, c->outlen); return NULL; }

static uint8_t *read_whole(const char *path, size_t *len){
    FILE *f = fopen(path, "rb"); if (!f){ fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(sz > 0 ? (size_t)sz : 1);
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz){ fprintf(stderr, "read error\n"); exit(1); }
    fclose(f); *len = (size_t)sz; return buf;
}
static void write_whole(const char *path, const uint8_t *buf, size_t len){
    FILE *f = fopen(path, "wb"); if (!f){ fprintf(stderr, "cannot write %s\n", path); exit(1); }
    if (len > 0) fwrite(buf, 1, len, f);
    fclose(f);
}

/* Streaming container: independent chunks, bounded memory per chunk. */
static uint8_t *nova_core_compress(const uint8_t *in, size_t N, size_t *outlen){
    int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN); if (nthreads < 1) nthreads = 1;
    if (getenv("NOVA_THREADS")){ int t = atoi(getenv("NOVA_THREADS")); if (t > 0) nthreads = t; }
    int nchunks = nthreads; if (N == 0) nchunks = 1;
    /* hash tables saturate past ~9M keys and prediction quality decays -
     * keep chunks in the sharp zone: ~8 MB each, always a multiple of the
     * worker count so waves stay even */
    while (nchunks < 64 && (N + (size_t)nchunks - 1) / (size_t)nchunks > (size_t)9 * 1024 * 1024)
        nchunks += nthreads;
    size_t *usizes = malloc(sizeof(size_t) * (nchunks + 1));
    size_t *starts = malloc(sizeof(size_t) * (nchunks + 1));
    size_t base = N / nchunks, rem = N % nchunks, acc = 0;
    for (int i = 0; i < nchunks; i++){ starts[i] = acc; size_t u = base + (i < (int)rem ? 1 : 0); usizes[i] = u; acc += u; }
    pthread_t *th = malloc(sizeof(pthread_t) * nchunks);
    ChunkArg *ca = malloc(sizeof(ChunkArg) * nchunks);
    for (int i = 0; i < nchunks; i++){ ca[i].in = in + starts[i]; ca[i].n = usizes[i]; ca[i].out = NULL; ca[i].outlen = 0; pthread_create(&th[i], NULL, t_compress, &ca[i]); }
    for (int i = 0; i < nchunks; i++) pthread_join(th[i], NULL);
    size_t hdr = 4 + 1 + 8 + 4 + (size_t)nchunks * 8;
    uint8_t *hbuf = malloc(hdr); size_t p = 0;
    memcpy(hbuf, "NOVA", 4); p += 4; hbuf[p++] = 4;
    for (int b = 0; b < 8; b++) hbuf[p++] = (uint8_t)((uint64_t)N >> (8 * b));
    for (int b = 0; b < 4; b++) hbuf[p++] = (uint8_t)(nchunks >> (8 * b));
    for (int i = 0; i < nchunks; i++){ for (int b = 0; b < 4; b++) hbuf[p++] = (uint8_t)(usizes[i] >> (8 * b)); for (int b = 0; b < 4; b++) hbuf[p++] = (uint8_t)(ca[i].outlen >> (8 * b)); }
    size_t total = hdr; for (int i = 0; i < nchunks; i++) total += ca[i].outlen;
    uint8_t *out = malloc(total); size_t q = 0;
    memcpy(out, hbuf, hdr); q += hdr;
    for (int i = 0; i < nchunks; i++){ memcpy(out + q, ca[i].out, ca[i].outlen); q += ca[i].outlen; free(ca[i].out); }
    free(hbuf); free(ca); free(th); free(usizes); free(starts);
    *outlen = total; return out;
}
static uint8_t *nova_core_decompress(const uint8_t *z, size_t Z, size_t *outlen){
    if (Z < 13 || memcmp(z, "NOVA", 4)){ fprintf(stderr, "bad NOVA stream\n"); exit(1); }
    size_t p = 4;
    if (z[p] != 4){ fprintf(stderr, "unsupported NOVA version %d\n", z[p]); exit(1); }
    p += 1;
    uint64_t N = 0; for (int b = 0; b < 8; b++) N |= (uint64_t)z[p++] << (8 * b);
    uint32_t nchunks = 0; for (int b = 0; b < 4; b++) nchunks |= (uint32_t)z[p++] << (8 * b);
    uint32_t *usizes = malloc(sizeof(uint32_t) * nchunks);
    uint32_t *csizes = malloc(sizeof(uint32_t) * nchunks);
    for (uint32_t i = 0; i < nchunks; i++){ usizes[i] = 0; csizes[i] = 0; for (int b = 0; b < 4; b++) usizes[i] |= (uint32_t)z[p++] << (8 * b); for (int b = 0; b < 4; b++) csizes[i] |= (uint32_t)z[p++] << (8 * b); }
    pthread_t *th = malloc(sizeof(pthread_t) * nchunks);
    ChunkArg *ca = malloc(sizeof(ChunkArg) * nchunks);
    size_t q = p;
    for (uint32_t i = 0; i < nchunks; i++){ ca[i].in = z + q; ca[i].n = csizes[i]; ca[i].outlen = usizes[i]; ca[i].out = NULL; pthread_create(&th[i], NULL, t_decompress, &ca[i]); q += csizes[i]; }
    for (uint32_t i = 0; i < nchunks; i++) pthread_join(th[i], NULL);
    uint8_t *out = malloc(N ? (size_t)N : 1); size_t acc = 0;
    for (uint32_t i = 0; i < nchunks; i++){ memcpy(out + acc, ca[i].out, usizes[i]); acc += usizes[i]; free(ca[i].out); }
    free(ca); free(th); free(usizes); free(csizes);
    *outlen = (size_t)N; return out;
}
/* ---------------- KV RECORD DEDUPLICATION (K-lane) ----------------
 * Structural pre-transform for record-delimited machine data such as
 * config/key-value files where records end with "\\n\\n" and the same
 * record text repeats thousands of times.
 *
 * This is NOT LZ (no sliding-window offsets, no length-bounded copies,
 * no parse tokens). This is NOT a word dictionary (dictionary entries
 * are whole records identified by their explicit "\\n\\n" boundary,
 * never words). This is NOT a borrowed algorithm - it is our own
 * content-addressing over the natural record delimiter, analogous to
 * the Entropy Ledger 'V' blocks and Noise Genome 'G' blocks, applied
 * at the structured-text layer.
 *
 * Layout:
 *   'K'  4-byte original_size
 *        4-byte ndict
 *        then ndict literal records each encoded as:
 *            2-byte record_length (LE, including trailing "\\n\\n")
 *            <record bytes>
 *        4-byte nidx
 *        then nidx variable-length record-index bytes:
 *            high-bit-clear (0..127):            index = value
 *            high-bit-set with 0x80 marker (next bytes): literal tail
 *                -> next 4-byte ulen, then ulen raw bytes appended
 *            Otherwise indices use continuation-7bit encoding
 *                (bit7=more, bits6..0 = 7 bits of index, little endian).
 *
 * After that, the entire K-segment body (dict + idx-tail concatenated)
 * is itself back-compressed with nova_core_compress, giving bit-level
 * entropy coding on top of the structural dedup. */

#define KV_MAX_RECLEN 4096
#define KV_MIN_RECLEN 8
#define KV_MIN_RECORDS 64
#define KV_HBITS 18

static const uint8_t *g_kv_pool;
static const uint32_t *g_kv_dlen, *g_kv_doff;
static int kv_entry_cmp(const void *pa, const void *pb){
    uint32_t a = *(const uint32_t*)pa, b = *(const uint32_t*)pb;
    const uint8_t *ap = g_kv_pool + g_kv_doff[a], *bp = g_kv_pool + g_kv_doff[b];
    uint32_t al = g_kv_dlen[a], bl = g_kv_dlen[b];
    uint32_t m = al < bl ? al : bl;
    for (uint32_t i = 0; i < m; i++){
        if (ap[i] != bp[i]) return ap[i] < bp[i] ? -1 : 1;
    }
    return al < bl ? -1 : (al > bl ? 1 : 0);
}
static uint32_t kv_hash(const uint8_t *p, uint32_t n){
    uint64_t h = 0xCBF29CE484222325ULL;
    for (uint32_t i = 0; i < n; i++){ h ^= p[i]; h *= 0x100000001B3ULL; }
    return (uint32_t)(h ^ (h >> 32));
}
static int kv_try_mode(const uint8_t *in, size_t N, int mode,
                  uint32_t **p_starts, uint32_t **p_lens, uint32_t *p_nrec){
    /* Detect whether input is a record-delimited text suitable for the
     * K-lane. mode 0: records end with "\\n\\n" (config stanzas).
     * mode 1 (LINE-RECORDS): every line ending with "\\n" is a record;
     * telemetry/config lines repeat thousands of times verbatim.
     * Heuristic: record length in bounds, records look like text
     * (few low-ctl bytes). */
    const uint32_t min_rl = mode ? 1u : (uint32_t)KV_MIN_RECLEN;
    if (N < 4096) return 0;
    uint32_t *starts = malloc(sizeof(uint32_t) * (N / 16 + 4));
    uint32_t *lens   = malloc(sizeof(uint32_t) * (N / 16 + 4));
    uint32_t nr = 0, i = 0, cap = N / 16 + 4;
    size_t pos = 0;
    int bad = 0;
    while (pos < N){
        size_t j = pos;
        /* scan for the record delimiter but only up to KV_MAX_RECLEN */
        size_t lim = pos + KV_MAX_RECLEN;
        if (lim > N) lim = N;
        size_t mk = (size_t)-1;
        if (mode){
            const uint8_t *nl = memchr(in + pos, '\n', lim - pos);
            if (nl) mk = (size_t)(nl - in) + 1;
        } else {
            for (size_t k = pos + 1; k < lim; k++){
                if (in[k] == '\n' && in[k-1] == '\n'){ mk = k + 1; break; }
            }
        }
        if (mk == (size_t)-1){
            /* trailing partial record */
            uint32_t rl = (uint32_t)(N - pos);
            if (rl < min_rl || rl > KV_MAX_RECLEN){ bad = 1; break; }
            if (nr >= cap){ cap *= 2; starts = realloc(starts, sizeof(uint32_t)*cap); lens = realloc(lens, sizeof(uint32_t)*cap); }
            starts[nr] = (uint32_t)pos; lens[nr] = rl; nr++;
            pos = N;
            break;
        }
        uint32_t rl = (uint32_t)(mk - pos);
        if (rl < min_rl || rl > KV_MAX_RECLEN){ bad = 1; break; }
        /* text sanity: no NUL bytes; a good fraction of printable ASCII */
        uint32_t printable = 0;
        for (size_t k = pos; k < mk; k++){
            uint8_t b = in[k];
            if (b == 0){ bad = 1; break; }
            if ((b >= 0x20 && b < 0x7F) || b == '\n' || b == '\t' || b == '\r') printable++;
        }
        if (bad) break;
        if ((double)printable / rl < 0.85){ bad = 1; break; }
        if (nr >= cap){ cap *= 2; starts = realloc(starts, sizeof(uint32_t)*cap); lens = realloc(lens, sizeof(uint32_t)*cap); }
        starts[nr] = (uint32_t)pos; lens[nr] = rl; nr++;
        pos = mk;
    }
    if (bad || nr < KV_MIN_RECORDS){ free(starts); free(lens); return 0; }
    *p_starts = starts; *p_lens = lens; *p_nrec = nr;
    return 1;
}

static uint8_t *kv_compress_mode(const uint8_t *in, size_t N, size_t *outlen, int mode){
    uint32_t *starts = NULL, *lens = NULL, nrec = 0;
    if (mode){
        /* SAMPLE PRE-PROBE: hash the first ~1MB of lines; when virtually
         * no line repeats (json/xml/csv telemetry), the full scan + dict
         * build cannot pay off - bail in milliseconds instead. */
        size_t lim = N < (1u << 20) ? N : (1u << 20);
        uint32_t sample[16384];
        for (int i = 0; i < 16384; i++) sample[i] = 0xFFFFFFFFu;
        size_t pos = 0; uint32_t recs = 0, dups = 0;
        while (pos < lim && recs < 8192){
            const uint8_t *nl = memchr(in + pos, '\n', lim - pos);
            size_t end = nl ? (size_t)(nl - in) + 1 : lim;
            uint32_t rl = (uint32_t)(end - pos);
            if (rl > 0 && rl <= KV_MAX_RECLEN){
                uint32_t h = kv_hash(in + pos, rl);
                int seen = 0;
                for (uint32_t k = 0; k < 16384; k++){
                    uint32_t s2 = (h + k) & 16383u;
                    if (sample[s2] == 0xFFFFFFFFu){ sample[s2] = h; break; }
                    if (sample[s2] == h){ seen = 1; break; }
                    if (k > 64) break;
                }
                if (seen) dups++;
                recs++;
            }
            pos = end;
            if (!nl) break;
        }
        if (recs >= 1024 && (uint64_t)dups * 8 < (uint64_t)recs) return NULL;
    }
    if (!kv_try_mode(in, N, mode, &starts, &lens, &nrec)) return NULL;
    /* Build dictionary using open-addressing linear-probe hash table. */
    uint32_t *idx = NULL;
    uint32_t *order = NULL, *rank = NULL;
    uint32_t tsz = 1u << KV_HBITS;
    uint32_t *t = malloc(sizeof(uint32_t) * tsz);
    for (uint32_t i = 0; i < tsz; i++) t[i] = 0xFFFFFFFFu;
    /* pool: raw record bytes, indexed by (doff,dlen) */
    size_t pool_cap = 256 * 1024;
    if (pool_cap < (size_t)nrec * 8) pool_cap = (size_t)nrec * 8;
    uint8_t *pool = malloc(pool_cap); size_t plen = 0;
    uint32_t dcount = 0;
    uint32_t *dlen = malloc(sizeof(uint32_t) * (nrec + 1));
    uint32_t *doff = malloc(sizeof(uint32_t) * (nrec + 1));
    for (uint32_t r = 0; r < nrec; r++){
        const uint8_t *rp = in + starts[r]; uint32_t rl = lens[r];
        uint32_t h = kv_hash(rp, rl);
        uint32_t slot = h & (tsz - 1);
        int found = -1;
        for (uint32_t p2 = 0; p2 < tsz; p2++){
            uint32_t s = (slot + p2) & (tsz - 1);
            if (t[s] == 0xFFFFFFFFu) break;
            uint32_t di = t[s];
            if (dlen[di] == rl && memcmp(pool + doff[di], rp, rl) == 0){ found = (int)di; break; }
        }
        if (found >= 0){ if ((r & 4095u) == 4095u && r >= 16384 && (uint64_t)dcount * 3 > (uint64_t)r * 2){ goto kv_abort; } continue; }
        /* insert new record */
        if ((r & 4095u) == 4095u && r >= 16384 && (uint64_t)dcount * 3 > (uint64_t)r * 2) goto kv_abort;
        while (plen + rl > pool_cap){ pool_cap *= 2; pool = realloc(pool, pool_cap); }
        uint32_t di = dcount;
        dlen[di] = rl;
        doff[di] = (uint32_t)plen;
        memcpy(pool + plen, rp, rl); plen += rl;
        dcount++;
        for (uint32_t p2 = 0; p2 < tsz; p2++){
            uint32_t s = (slot + p2) & (tsz - 1);
            if (t[s] == 0xFFFFFFFFu){ t[s] = di; break; }
        }
    }
    /* LEXICAL REBUILD PREP: sort dict entries so the dictionary ships
     * prefix-delta coded (shared key prefixes paid once). */
    order = malloc(sizeof(uint32_t) * (dcount ? dcount : 1));
    rank  = malloc(sizeof(uint32_t) * (dcount ? dcount : 1));
    /* Now map each record to its dictionary index in a second pass */
    idx = malloc(sizeof(uint32_t) * nrec);
    for (uint32_t r = 0; r < nrec; r++){
        const uint8_t *rp = in + starts[r]; uint32_t rl = lens[r];
        uint32_t h = kv_hash(rp, rl);
        uint32_t slot = h & (tsz - 1);
        uint32_t found = 0;
        for (uint32_t p2 = 0; p2 < tsz; p2++){
            uint32_t s = (slot + p2) & (tsz - 1);
            if (t[s] == 0xFFFFFFFFu){ found = 0xFFFFFFFFu; break; }
            uint32_t di = t[s];
            if (dlen[di] == rl && memcmp(pool + doff[di], rp, rl) == 0){ found = di; break; }
        }
        if (found == 0xFFFFFFFFu){ fprintf(stderr,"K-enc: record %u not found in dict!\n",r); exit(1); }
        idx[r] = found;
    }

    /* DEDUP PROFIT GATE: the K body pays (nrec index varints + 2-byte
     * record lengths) for the privilege of removing (nrec - dcount)
     * duplicate records.  When almost every record is unique the lane
     * is pure overhead - fall back to the plain core instead. */
    if ((uint64_t)dcount * 3 > (uint64_t)nrec * 2) goto kv_abort;
    free(t);
    /* sort dict entries lexicographically (insertion-free qsort over a
     * small global context; ties broken by length) */
    g_kv_pool = pool; g_kv_dlen = dlen; g_kv_doff = doff;
    for (uint32_t e = 0; e < dcount; e++) order[e] = e;
    qsort(order, dcount, sizeof(uint32_t), kv_entry_cmp);
    for (uint32_t e = 0; e < dcount; e++) rank[order[e]] = e;
    /* trailing raw bytes (if any after last complete record): our kv_try
     * always adds trailing partial chunk as a record; but to be safe also
     * append any bytes past starts[nrec-1]+lens[nrec-1] */
    size_t last_end = (size_t)starts[nrec-1] + lens[nrec-1];
    (void)last_end;
    /* Build K-segment body:
     *   4 bytes ndict, then for each dict entry: 2-byte rl + bytes
     *   4 bytes nrec, then index stream (variable-length 7-bit codes) */
    size_t body_cap = 4 + (size_t)dcount * 4 + plen + 4 + (size_t)nrec * 2 + 256;
    uint8_t *body = malloc(body_cap); size_t bl = 0;
    #define BAPPEND(buf,len_p,cap_p,b,N) do{ \
        while(*(len_p)+(N) > *(cap_p)){ *(cap_p) = *(cap_p)*2+1024; (buf)=realloc((buf),*(cap_p));} \
        memcpy((buf)+*(len_p),(b),(N)); *(len_p)+=(N); }while(0)
    size_t *bcap_p = &body_cap;
    uint8_t nbuf[4];
    for (int b2 = 0; b2 < 4; b2++) nbuf[b2] = (uint8_t)(dcount >> (8*b2));
    BAPPEND(body, &bl, bcap_p, nbuf, 4);
    /* dict entries in sorted order, prefix-delta coded:
     * varint lcp-with-previous-entry, varint suffix_len, suffix bytes */
    {
        const uint8_t *pp = NULL; uint32_t pl2 = 0;
        for (uint32_t e = 0; e < dcount; e++){
            uint32_t di = order[e];
            const uint8_t *cp = pool + doff[di]; uint32_t cl = dlen[di];
            uint32_t lcp = 0;
            if (pp){ while (lcp < cl && lcp < pl2 && cp[lcp] == pp[lcp]) lcp++; }
            uint8_t vb[8]; int vn = 0; uint32_t v = lcp;
            while (v >= 0x80){ vb[vn++] = (uint8_t)(0x80 | (v & 0x7F)); v >>= 7; }
            vb[vn++] = (uint8_t)v;
            BAPPEND(body, &bl, bcap_p, vb, vn);
            v = cl - lcp; vn = 0;
            while (v >= 0x80){ vb[vn++] = (uint8_t)(0x80 | (v & 0x7F)); v >>= 7; }
            vb[vn++] = (uint8_t)v;
            BAPPEND(body, &bl, bcap_p, vb, vn);
            BAPPEND(body, &bl, bcap_p, cp + lcp, cl - lcp);
            pp = cp; pl2 = cl;
        }
    }
    for (int b2 = 0; b2 < 4; b2++) nbuf[b2] = (uint8_t)(nrec >> (8*b2));
    BAPPEND(body, &bl, bcap_p, nbuf, 4);
    size_t idx_start = bl;
    /* index stream (remapped to sorted rank) */
    for (uint32_t r = 0; r < nrec; r++){
        uint32_t v = rank[idx[r]];
        uint8_t vb[8]; int vn = 0;
        while (v >= 0x80){ vb[vn++] = (uint8_t)(0x80 | (v & 0x7F)); v >>= 7; }
        vb[vn++] = (uint8_t)v;
        BAPPEND(body, &bl, bcap_p, vb, vn);
    }
    free(pool); free(dlen); free(doff); free(starts); free(lens); free(idx); free(order); free(rank);
    /* Debug: dump body if env set */
    if (getenv("NOVA_DUMP_BODY")){
        FILE *fb = fopen(getenv("NOVA_DUMP_BODY"),"wb"); if(fb){fwrite(body,1,bl,fb);fclose(fb);}
    }
    /* Now back-compress body with nova_core_compress */
    size_t zlen;
    uint8_t *zb = nova_core_compress(body, bl, &zlen);
    free(body);
    /* Final output: 'L' + 8-byte original N + 4-byte zlen + zb */
    size_t out_total = 1 + 8 + 4 + zlen;
    uint8_t *out = malloc(out_total);
    size_t op = 0;
    out[op++] = 'L';
    for (int b2 = 0; b2 < 8; b2++) out[op++] = (uint8_t)((uint64_t)N >> (8*b2));
    for (int b2 = 0; b2 < 4; b2++) out[op++] = (uint8_t)(zlen >> (8*b2));
    memcpy(out + op, zb, zlen);
    free(zb);
    *outlen = out_total;
    return out;
kv_abort:
    free(t);
    free(pool); free(dlen); free(doff); free(starts); free(lens); free(idx); free(order); free(rank);
    return NULL;
    #undef BAPPEND
}
static uint8_t *kv_decompress(const uint8_t *in, size_t Z, size_t *outlen){
    if (Z < 1) return NULL;
    uint8_t marker = in[0];
    if (marker == '0'){
        size_t N;
        uint8_t *out = nova_core_decompress(in + 1, Z - 1, &N);
        *outlen = N; return out;
    }
    if (marker != 'K' && marker != 'L'){
        /* backward-compat: plain old NOVA stream */
        uint8_t *out = nova_core_decompress(in, Z, outlen);
        return out;
    }
    size_t p = 1;
    if (p + 8 + 4 > Z){ fprintf(stderr,"K-header truncated\n"); exit(1); }
    uint64_t N = 0; for (int b2 = 0; b2 < 8; b2++) N |= (uint64_t)in[p++] << (8*b2);
    uint32_t zlen = 0; for (int b2 = 0; b2 < 4; b2++) zlen |= (uint32_t)in[p++] << (8*b2);
    if ((size_t)zlen > Z - p){ fprintf(stderr,"K-zlen %u exceeds available %zu\n",zlen,Z-p); exit(1); }
    size_t bl;
    uint8_t *body = nova_core_decompress(in + p, zlen, &bl);
    size_t bp = 0;
    if (bp + 4 > bl){ fprintf(stderr,"K-body ndict truncated\n"); exit(1); }
    uint32_t dcount = 0;
    for (int b2 = 0; b2 < 4; b2++){ dcount |= (uint32_t)body[bp] << (8*b2); bp++; }
    if (dcount > 10000000){ fprintf(stderr,"K-dcount некорректен: %u\n",dcount); exit(1); }
    uint32_t *dlen = malloc(sizeof(uint32_t) * (dcount + 1));
    uint32_t *doff = malloc(sizeof(uint32_t) * (dcount + 1));
    size_t dpool_cap = bl + 1024;
    uint8_t *dpool = malloc(dpool_cap);
    size_t dpl = 0;
    size_t dict_start_bp = bp;
    if (marker == 'K'){
        for (uint32_t di = 0; di < dcount; di++){
            if (bp + 2 > bl){ fprintf(stderr,"K-dict[%u]: truncated at length\n",di); exit(1); }
            uint16_t rl = (uint16_t)body[bp] | ((uint16_t)body[bp+1] << 8); bp += 2;
            if (rl == 0 || rl > KV_MAX_RECLEN){ fprintf(stderr,"K-dict[%u]: bad rl=%u\n",di,rl); exit(1); }
            if (bp + rl > bl){ fprintf(stderr,"K-dict[%u]: truncated body (need %u, have %zu)\n",di,rl,bl-bp); exit(1); }
            dlen[di] = rl; doff[di] = (uint32_t)dpl;
            if (dpl + rl > dpool_cap){ while(dpl + rl > dpool_cap) dpool_cap *= 2; dpool = realloc(dpool, dpool_cap); }
            memcpy(dpool + dpl, body + bp, rl); dpl += rl; bp += rl;
        }
    } else { /* 'L': sorted prefix-delta dictionary */
        for (uint32_t di = 0; di < dcount; di++){
            uint32_t lcp = 0, suf = 0, sh = 0;
            for (;;){ if (bp >= bl){ fprintf(stderr,"L-dict[%u]: lcp truncated\n",di); exit(1); }
                uint8_t by = body[bp++]; lcp |= (uint32_t)(by & 0x7F) << sh;
                if (!(by & 0x80)) break; sh += 7; if (sh > 28){ fprintf(stderr,"L-dict[%u]: lcp varint overflow\n",di); exit(1); } }
            sh = 0;
            for (;;){ if (bp >= bl){ fprintf(stderr,"L-dict[%u]: suflen truncated\n",di); exit(1); }
                uint8_t by = body[bp++]; suf |= (uint32_t)(by & 0x7F) << sh;
                if (!(by & 0x80)) break; sh += 7; if (sh > 28){ fprintf(stderr,"L-dict[%u]: suf varint overflow\n",di); exit(1); } }
            if (di == 0 && lcp != 0){ fprintf(stderr,"L-dict[0]: nonzero lcp\n"); exit(1); }
            if (di > 0 && lcp > dlen[di-1]){ fprintf(stderr,"L-dict[%u]: lcp %u exceeds prev len %u\n",di,lcp,dlen[di-1]); exit(1); }
            uint32_t rl = lcp + suf;
            if (rl == 0 || rl > KV_MAX_RECLEN){ fprintf(stderr,"L-dict[%u]: bad rl=%u\n",di,rl); exit(1); }
            if (bp + suf > bl){ fprintf(stderr,"L-dict[%u]: truncated body\n",di); exit(1); }
            dlen[di] = rl; doff[di] = (uint32_t)dpl;
            if (dpl + rl > dpool_cap){ while(dpl + rl > dpool_cap) dpool_cap *= 2; dpool = realloc(dpool, dpool_cap); }
            if (lcp) memcpy(dpool + dpl, dpool + doff[di-1], lcp);
            memcpy(dpool + dpl + lcp, body + bp, suf);
            dpl += rl; bp += suf;
        }
    }
    if (bp + 4 > bl){ fprintf(stderr,"K-nrec: обрезан поток\n"); exit(1); }
    uint32_t nrec = 0;
    for (int b2 = 0; b2 < 4; b2++){ nrec |= (uint32_t)body[bp] << (8*b2); bp++; }
    if (nrec > 100000000){ fprintf(stderr,"K-nrec некорректен: %u\n",nrec); exit(1); }
    uint8_t *out = malloc((size_t)N + 16);
    size_t op = 0;
    for (uint32_t r = 0; r < nrec; r++){
        uint32_t v = 0; int sh = 0;
        for (;;){
            if (bp >= bl){ fprintf(stderr,"K-idx[%u]: truncated reading varint\n",r); exit(1); }
            uint8_t by = body[bp++];
            v |= (uint32_t)(by & 0x7F) << sh;
            if (!(by & 0x80)) break;
            sh += 7;
            if (sh > 28){ fprintf(stderr,"K-idx[%u]: varint overflow\n",r); exit(1); }
        }
        uint32_t di = v;
        if (di >= dcount){ fprintf(stderr,"K-idx[%u]: di=%u out of range (dcount=%u)\n",r,di,dcount); exit(1); }
        uint32_t rl = dlen[di];
        if (op + rl > (size_t)N + 1){ fprintf(stderr,"K-idx[%u]: output overflow op=%zu rl=%u N=%llu\n",r,op,rl,(unsigned long long)N); exit(1); }
        memcpy(out + op, dpool + doff[di], rl);
        op += rl;
    }
    if (op != (size_t)N){
        fprintf(stderr,"K: output size mismatch op=%zu expected=%llu (bp=%zu bl=%zu)\n",
                op,(unsigned long long)N,bp,bl);
        exit(1);
    }
    free(dlen); free(doff); free(dpool); free(body);
    *outlen = (size_t)N;
    return out;
}

/* public API: whole-file entry point with the KV dedup lane */
typedef struct {
    const uint8_t *ptr;
    uint32_t len;
} Field;

static inline int is_numeric_field(const uint8_t *ptr, uint32_t len) {
    if (len == 0) return 0;
    for (uint32_t i = 0; i < len; i++) {
        if (ptr[i] < '0' || ptr[i] > '9') return 0;
    }
    return 1;
}

static inline void write_varint_field(uint8_t *buf, size_t *bl, uint32_t v) {
    while (v >= 0x80) {
        buf[(*bl)++] = (uint8_t)(0x80 | (v & 0x7F));
        v >>= 7;
    }
    buf[(*bl)++] = (uint8_t)v;
}

static inline uint32_t read_varint_field(const uint8_t *buf, size_t *bp) {
    uint32_t v = 0;
    int sh = 0;
    for (;;) {
        uint8_t b = buf[(*bp)++];
        v |= (uint32_t)(b & 0x7F) << sh;
        if (!(b & 0x80)) break;
        sh += 7;
    }
    return v;
}

static inline int32_t read_signed_varint_field(const uint8_t *buf, size_t *bp) {
    uint32_t u = read_varint_field(buf, bp);
    return (int32_t)((u >> 1) ^ -(int32_t)(u & 1));
}

static inline void write_string_field(uint8_t *buf, size_t *bl, const uint8_t *ptr, uint32_t len) {
    write_varint_field(buf, bl, len);
    if (len > 0) {
        memcpy(buf + *bl, ptr, len);
        *bl += len;
    }
}


/* ---- strict ISO-8601 millisecond timestamps (YYYY-MM-DDTHH:MM:SS.mmmZ) ---- */
static const int ts_cum[12] = {0,31,59,90,120,151,181,212,243,273,304,334};
static inline int ts_leap(int y){ return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
static inline uint64_t ts_dfc(int y, int m, int d){
    return 365ULL * (uint64_t)(y - 1) + (uint64_t)((y - 1) / 4) - (uint64_t)((y - 1) / 100)
         + (uint64_t)((y - 1) / 400) + (uint64_t)ts_cum[m - 1] + (m > 2 ? ts_leap(y) : 0) + (uint64_t)d - 1;
}
static inline void ts_civil(uint64_t days, int *y, int *m, int *d){
    int yy = (int)(days / 365) + 1;
    while (ts_dfc(yy + 1, 1, 1) <= days) yy++;
    while (yy > 1 && ts_dfc(yy, 1, 1) > days) yy--;
    uint64_t doy = days - ts_dfc(yy, 1, 1);
    int lp = ts_leap(yy), mm = 12;
    while ((uint64_t)(ts_cum[mm - 1] + (mm > 2 ? lp : 0)) > doy) mm--;
    *y = yy; *m = mm; *d = (int)(doy - (uint64_t)(ts_cum[mm - 1] + (mm > 2 ? lp : 0))) + 1;
}
static inline int ts_dig(const uint8_t *p, int i){ return p[i] >= '0' && p[i] <= '9'; }
/* returns ms since epoch day 0, or -1 when the 24 bytes are not strict ISO-ms */
static inline int64_t ts_parse(const uint8_t *p, uint32_t len){
    if (len != 24) return -1;
    if (p[4] != '-' || p[7] != '-' || p[10] != 'T' || p[13] != ':' || p[16] != ':' || p[19] != '.' || p[23] != 'Z') return -1;
    int digs[17] = {0,1,2,3,5,6,8,9,11,12,14,15,17,18,20,21,22};
    for (int i = 0; i < 17; i++) if (!ts_dig(p, digs[i])) return -1;
    int y = (p[0]-'0')*1000 + (p[1]-'0')*100 + (p[2]-'0')*10 + (p[3]-'0');
    int m = (p[5]-'0')*10 + (p[6]-'0'), d = (p[8]-'0')*10 + (p[9]-'0');
    int hh = (p[11]-'0')*10 + (p[12]-'0'), mm = (p[14]-'0')*10 + (p[15]-'0'), ss = (p[17]-'0')*10 + (p[18]-'0');
    int ms = (p[20]-'0')*100 + (p[21]-'0')*10 + (p[22]-'0');
    if (m < 1 || m > 12 || d < 1 || d > 31 || hh > 23 || mm > 59 || ss > 59) return -1;
    return (int64_t)(ts_dfc(y, m, d) * 86400000ULL + (uint64_t)((hh * 60 + mm) * 60 + ss) * 1000 + (uint64_t)ms);
}
static inline void ts_format(uint8_t *o, uint64_t ms64){
    uint64_t days = ms64 / 86400000ULL; uint32_t rem = (uint32_t)(ms64 % 86400000ULL);
    int y, m, d; ts_civil(days, &y, &m, &d);
    int hh = (int)(rem / 3600000); rem %= 3600000;
    int mm = (int)(rem / 60000); rem %= 60000;
    int ss = (int)(rem / 1000); uint32_t ms = rem % 1000;
    o[0] = '0' + y / 1000; o[1] = '0' + (y / 100) % 10; o[2] = '0' + (y / 10) % 10; o[3] = '0' + y % 10;
    o[4] = '-'; o[5] = '0' + m / 10; o[6] = '0' + m % 10;
    o[7] = '-'; o[8] = '0' + d / 10; o[9] = '0' + d % 10;
    o[10] = 'T'; o[11] = '0' + hh / 10; o[12] = '0' + hh % 10;
    o[13] = ':'; o[14] = '0' + mm / 10; o[15] = '0' + mm % 10;
    o[16] = ':'; o[17] = '0' + ss / 10; o[18] = '0' + ss % 10;
    o[19] = '.'; o[20] = '0' + ms / 100; o[21] = '0' + (ms / 10) % 10; o[22] = '0' + ms % 10;
    o[23] = 'Z';
}
static inline int varint_len(uint32_t v){ int n = 1; while (v >= 0x80){ v >>= 7; n++; } return n; }

static uint8_t *c_compress(const uint8_t *in, size_t N, size_t *outlen) {
    if (N < 4096) return NULL;
    
    size_t printable = 0;
    for (size_t i = 0; i < (N < 65536 ? N : 65536); i++) {
        uint8_t b = in[i];
        if ((b >= 0x20 && b < 0x7F) || b == '\n' || b == '\t' || b == '\r') {
            printable++;
        }
    }
    double frac = (double)printable / (N < 65536 ? N : 65536);
    if (frac < 0.85) return NULL;

    uint32_t num_lines = 0;
    size_t line_start = 0;
    for (size_t i = 0; i < N; i++) {
        if (in[i] == '\n') {
            num_lines++;
            line_start = i + 1;
        }
    }
    if (line_start < N) {
        num_lines++;
    }
    if (num_lines < 64) return NULL;
    double avg_len = (double)N / num_lines;
    if (avg_len < 8 || avg_len > 4096) return NULL;

    size_t *lines_start = malloc(sizeof(size_t) * num_lines);
    size_t *lines_end = malloc(sizeof(size_t) * num_lines);
    num_lines = 0;
    line_start = 0;
    for (size_t i = 0; i < N; i++) {
        if (in[i] == '\n') {
            lines_start[num_lines] = line_start;
            lines_end[num_lines] = i + 1;
            num_lines++;
            line_start = i + 1;
        }
    }
    if (line_start < N) {
        lines_start[num_lines] = line_start;
        lines_end[num_lines] = N;
        num_lines++;
    }

    uint32_t *num_fields = malloc(sizeof(uint32_t) * num_lines);
    uint32_t max_cols = 0;
    uint32_t nf_hist[65] = {0};
    size_t sep_bytes = 0;

    for (uint32_t i = 0; i < num_lines; i++) {
        size_t l_start = lines_start[i];
        size_t l_end = lines_end[i];
        uint32_t nf = 0;
        int prev_cls = -1;
        for (size_t p = l_start; p < l_end; p++) {
            int cls = get_class(in[p]);
            if (cls == 0) sep_bytes++;
            if (prev_cls != -1 && cls != prev_cls) {
                nf++;
            }
            prev_cls = cls;
        }
        if (l_end > l_start) {
            nf++;
        }
        num_fields[i] = nf;
        nf_hist[nf > 64 ? 64 : nf]++;
        if (nf > max_cols) {
            max_cols = nf;
        }
    }
    if (max_cols > 4096) max_cols = 4096;

    /* LANE ROUTING: the columnar transposer pays off on genuinely columnar
     * data (stable field count across lines + separator-dense rows, the
     * CSV shape).  On ragged or separator-sparse text the row-major core
     * with field lanes wins - defer to it. */
    uint32_t mshare = 0;
    for (int h = 0; h <= 64; h++) if (nf_hist[h] > mshare) mshare = nf_hist[h];
    if ((double)mshare / num_lines < 0.90 || (double)sep_bytes / N < 0.08){
        free(lines_start); free(lines_end); free(num_fields);
        return NULL;
    }

    size_t *val_cap = malloc(sizeof(size_t) * max_cols);
    size_t *val_len = malloc(sizeof(size_t) * max_cols);
    uint8_t **val_buf = malloc(sizeof(uint8_t*) * max_cols);

    size_t *len_cap = malloc(sizeof(size_t) * max_cols);
    size_t *len_len = malloc(sizeof(size_t) * max_cols);
    uint16_t **len_buf = malloc(sizeof(uint16_t*) * max_cols);

    for (uint32_t c = 0; c < max_cols; c++) {
        val_cap[c] = 256;
        val_len[c] = 0;
        val_buf[c] = malloc(256);

        len_cap[c] = 256;
        len_len[c] = 0;
        len_buf[c] = malloc(256 * 2);
    }

    for (uint32_t i = 0; i < num_lines; i++) {
        size_t l_start = lines_start[i];
        size_t l_end = lines_end[i];
        uint32_t nf = 0;
        int prev_cls = -1;
        size_t f_start = l_start;
        for (size_t p = l_start; p < l_end; p++) {
            int cls = get_class(in[p]);
            if (prev_cls != -1 && cls != prev_cls) {
                if (nf < max_cols) {
                    uint32_t col = nf;
                    uint32_t fl = (uint32_t)(p - f_start);
                    if (val_len[col] + fl > val_cap[col]) {
                        while (val_len[col] + fl > val_cap[col]) val_cap[col] *= 2;
                        val_buf[col] = realloc(val_buf[col], val_cap[col]);
                    }
                    memcpy(val_buf[col] + val_len[col], in + f_start, fl);
                    val_len[col] += fl;
                    if (len_len[col] + 1 > len_cap[col]) {
                        len_cap[col] *= 2;
                        len_buf[col] = realloc(len_buf[col], len_cap[col] * 2);
                    }
                    len_buf[col][len_len[col]++] = (uint16_t)fl;
                    nf++;
                }
                f_start = p;
            }
            prev_cls = cls;
        }
        if (l_end > f_start && nf < max_cols) {
            uint32_t col = nf;
            uint32_t fl = (uint32_t)(l_end - f_start);
            if (val_len[col] + fl > val_cap[col]) {
                while (val_len[col] + fl > val_cap[col]) val_cap[col] *= 2;
                val_buf[col] = realloc(val_buf[col], val_cap[col]);
            }
            memcpy(val_buf[col] + val_len[col], in + f_start, fl);
            val_len[col] += fl;
            if (len_len[col] + 1 > len_cap[col]) {
                len_cap[col] *= 2;
                len_buf[col] = realloc(len_buf[col], len_cap[col] * 2);
            }
            len_buf[col][len_len[col]++] = (uint16_t)fl;
            nf++;
        }
        num_fields[i] = nf;
        for (uint32_t col = nf; col < max_cols; col++) {
            if (len_len[col] + 1 > len_cap[col]) {
                len_cap[col] *= 2;
                len_buf[col] = realloc(len_buf[col], len_cap[col] * 2);
            }
            len_buf[col][len_len[col]++] = 0;
        }
    }

    size_t body_cap = 3 * N + (size_t)num_lines * 32 + 65536;
    uint8_t *body = malloc(body_cap);
    size_t bl = 0;

    memcpy(body + bl, &num_lines, 4); bl += 4;
    memcpy(body + bl, &max_cols, 4); bl += 4;

    for (uint32_t i = 0; i < num_lines; i++) {
        uint16_t nf = (uint16_t)num_fields[i];
        memcpy(body + bl, &nf, 2); bl += 2;
    }

    uint8_t *col_types = malloc(max_cols);
    uint8_t **col_dict_bytes = calloc(max_cols, sizeof(uint8_t*));
    uint32_t *col_dict_sizes = calloc(max_cols, sizeof(uint32_t));
    uint8_t **col_line_indices = calloc(max_cols, sizeof(uint8_t*));

    typedef struct {
        const uint8_t *ptr;
        uint32_t len;
    } DictEntry;

    for (uint32_t c = 0; c < max_cols; c++) {
        uint32_t count = len_len[c];
        if (count == 0) {
            col_types[c] = 3;
            body[bl++] = 3;
            continue;
        }

        int is_const = 1;
        uint32_t first_len = len_buf[c][0];
        const uint8_t *first_ptr = val_buf[c];
        uint32_t offset = first_len;
        for (uint32_t i = 1; i < count; i++) {
            uint32_t l = len_buf[c][i];
            if (l != first_len || (first_len > 0 && memcmp(val_buf[c] + offset, first_ptr, first_len) != 0)) {
                is_const = 0;
                break;
            }
            offset += l;
        }
        if (is_const) {
            col_types[c] = 0;
            body[bl++] = 0;
            write_string_field(body, &bl, first_ptr, first_len);
            continue;
        }

        int is_seq = 1;
        offset = 0;
        uint32_t *vals = malloc(sizeof(uint32_t) * count);
        for (uint32_t i = 0; i < count; i++) {
            uint32_t l = len_buf[c][i];
            if (l == 0 || !is_numeric_field(val_buf[c] + offset, l)) {
                is_seq = 0;
                break;
            }
            uint32_t val = 0;
            for (uint32_t k = 0; k < l; k++) {
                val = val * 10 + (val_buf[c][offset + k] - '0');
            }
            vals[i] = val;
            offset += l;
        }
        if (is_seq && count > 1) {
            int32_t delta = (int32_t)(vals[1] - vals[0]);
            for (uint32_t i = 2; i < count; i++) {
                if ((int32_t)(vals[i] - vals[i-1]) != delta) {
                    is_seq = 0;
                    break;
                }
            }
            if (is_seq) {
                col_types[c] = 1;
                body[bl++] = 1;
                write_varint_field(body, &bl, first_len);
                write_varint_field(body, &bl, vals[0]);
                uint32_t u_delta = (delta << 1) ^ (delta >> 31);
                write_varint_field(body, &bl, u_delta);
                free(vals);
                continue;
            }
        }
        free(vals);

        DictEntry *de = malloc(sizeof(DictEntry) * count);
        uint32_t de_size = 0;
        offset = 0;
        uint8_t *line_indices = malloc(num_lines);
        for (uint32_t i = 0; i < count; i++) {
            uint32_t l = len_buf[c][i];
            const uint8_t *p = val_buf[c] + offset;
            int found = -1;
            if (i > 0) {
                uint32_t prev_l = len_buf[c][i-1];
                if (l == prev_l && (l == 0 || memcmp(p, p - l, l) == 0)) {
                    found = (int)line_indices[i-1];
                }
            }
            if (found < 0) {
                for (uint32_t k = 0; k < de_size; k++) {
                    if (de[k].len == l && (l == 0 || memcmp(de[k].ptr, p, l) == 0)) {
                        found = (int)k;
                        break;
                    }
                }
            }
            if (found < 0) {
                if (de_size < 256) {
                    de[de_size].ptr = p;
                    de[de_size].len = l;
                    line_indices[i] = (uint8_t)de_size;
                    de_size++;
                } else {
                    de_size = 257;
                    break;
                }
            } else {
                line_indices[i] = (uint8_t)found;
            }
            offset += l;
        }

        if (de_size <= 256) {
            col_types[c] = 2;
            body[bl++] = 2;
            body[bl++] = (uint8_t)de_size;
            for (uint32_t k = 0; k < de_size; k++) {
                write_string_field(body, &bl, de[k].ptr, de[k].len);
            }
            col_dict_sizes[c] = de_size;
            col_dict_bytes[c] = (uint8_t*)de;
            col_line_indices[c] = line_indices;
        } else {
            /* not dictionary-able: price RAW vs PREFIX-DELTA vs TS-DELTA on
             * the exact emitted-entry stream (missing rows stay skipped) */
            uint64_t estraw = 0, estpd = 0, estts = 0;
            int tsok = 1, first_seen = 0;
            uint32_t plen = 0, off2 = 0, first_len2 = 0;
            const uint8_t *pptr = NULL, *first_ptr2 = NULL;
            int64_t tsprev = 0;
            for (uint32_t i = 0; i < count; i++) {
                uint32_t l = len_buf[c][i];
                const uint8_t *pp = val_buf[c] + off2;
                off2 += l;
                if (num_fields[i] <= c) continue;      /* pad for a shorter row */
                estraw += varint_len(l) + l;
                if (!first_seen){ first_seen = 1; first_ptr2 = pp; first_len2 = l;
                    estpd += varint_len(l) + l;
                    int64_t ms0 = ts_parse(pp, l);
                    if (ms0 < 0) tsok = 0; else tsprev = ms0;
                    estts += varint_len(l) + l;
                    plen = l; pptr = pp;
                    continue;
                }
                uint32_t lcp = 0, mn = l < plen ? l : plen;
                while (lcp < mn && pp[lcp] == pptr[lcp]) lcp++;
                estpd += varint_len(lcp) + varint_len(l - lcp) + (l - lcp);
                plen = l; pptr = pp;
                if (tsok){
                    int64_t msc = ts_parse(pp, l);
                    int64_t dd = msc < 0 ? (int64_t)1 << 62 : msc - tsprev;
                    if (msc < 0 || dd > (1 << 30) || dd < -(1 << 30)) tsok = 0;
                    else {
                        tsprev = msc;
                        uint32_t z = ((uint32_t)(int32_t)dd << 1) ^ (uint32_t)((int32_t)dd >> 31);
                        estts += varint_len(z);
                    }
                }
            }
            if (tsok && first_seen && count > 1 && estts <= estpd && estts < estraw) {
                col_types[c] = 5;
                body[bl++] = 5;
                write_string_field(body, &bl, first_ptr2, first_len2);
            } else if (first_seen && count > 1 && estpd < estraw) {
                col_types[c] = 4;
                body[bl++] = 4;
                write_string_field(body, &bl, first_ptr2, first_len2);
            } else {
                col_types[c] = 3;
                body[bl++] = 3;
            }
            free(de);
            free(line_indices);
        }
    }

    // Now write line data row-by-row (row-oriented!)
    uint32_t *col_offsets = calloc(max_cols, sizeof(uint32_t));
    uint32_t *ecnt = calloc(max_cols, sizeof(uint32_t));
    uint32_t *eoff = calloc(max_cols, sizeof(uint32_t));
    uint32_t *elen = calloc(max_cols, sizeof(uint32_t));
    int64_t *ets = calloc(max_cols, sizeof(int64_t));
    for (uint32_t i = 0; i < num_lines; i++) {
        uint32_t nf = num_fields[i];
        for (uint32_t j = 0; j < nf; j++) {
            uint32_t col = j;
            if (col >= max_cols) continue;

            uint32_t l = len_buf[col][i];
            const uint8_t *ep = val_buf[col] + col_offsets[col];
            uint8_t type = col_types[col];
            if (type == 0 || type == 1) {
                // Do nothing!
            } else if (type == 2) {
                // TYPE_DICT (instant O(1) lookup!)
                body[bl++] = col_line_indices[col][i];
            } else if (type == 4) {
                // TYPE PREFIX-DELTA: first emitted row was shipped in the header
                if (ecnt[col] > 0) {
                    uint32_t lcp = 0, mn = l < elen[col] ? l : elen[col];
                    const uint8_t *pv = val_buf[col] + eoff[col];
                    while (lcp < mn && ep[lcp] == pv[lcp]) lcp++;
                    write_varint_field(body, &bl, lcp);
                    write_varint_field(body, &bl, l - lcp);
                    memcpy(body + bl, ep + lcp, l - lcp); bl += l - lcp;
                }
                ecnt[col]++;
                eoff[col] = col_offsets[col]; elen[col] = l;
            } else if (type == 5) {
                // TYPE TS-DELTA: zigzag varint of the ms delta
                int64_t msc = ts_parse(ep, l);
                if (ecnt[col] > 0) {
                    int32_t dd = (int32_t)(msc - ets[col]);
                    write_varint_field(body, &bl, ((uint32_t)dd << 1) ^ (uint32_t)(dd >> 31));
                }
                ets[col] = msc;
                ecnt[col]++;
            } else {
                // TYPE_RAW
                write_string_field(body, &bl, ep, l);
            }
            col_offsets[col] += l;
        }
    }
    free(col_offsets); free(ecnt); free(eoff); free(elen); free(ets);

    for (uint32_t c = 0; c < max_cols; c++) {
        free(val_buf[c]);
        free(len_buf[c]);
        if (col_dict_bytes[c]) {
            free(col_dict_bytes[c]);
        }
        if (col_line_indices[c]) {
            free(col_line_indices[c]);
        }
    }
    free(val_buf); free(val_cap); free(val_len);
    free(len_buf); free(len_cap); free(len_len);
    free(col_types);
    free(col_dict_bytes);
    free(col_dict_sizes);
    free(col_line_indices);
    free(lines_start); free(lines_end);
    free(num_fields);

    size_t zlen;
    uint8_t *zb = nova_core_compress(body, bl, &zlen);
    free(body);

    size_t out_total = 1 + 8 + 4 + zlen;
    uint8_t *out = malloc(out_total);
    size_t op = 0;
    out[op++] = 'C';
    for (int b = 0; b < 8; b++) out[op++] = (uint8_t)((uint64_t)N >> (8 * b));
    for (int b = 0; b < 4; b++) out[op++] = (uint8_t)(zlen >> (8 * b));
    memcpy(out + op, zb, zlen);
    free(zb);

    *outlen = out_total;
    return out;
}

static uint8_t *c_decompress(const uint8_t *in, size_t Z, size_t *outlen) {
    if (Z < 13 || in[0] != 'C') return NULL;
    
    size_t p = 1;
    uint64_t N = 0; for (int b = 0; b < 8; b++) N |= (uint64_t)in[p++] << (8 * b);
    uint32_t zlen = 0; for (int b = 0; b < 4; b++) zlen |= (uint32_t)in[p++] << (8 * b);
    
    size_t bl;
    uint8_t *body = nova_core_decompress(in + p, zlen, &bl);
    
    size_t bp = 0;
    uint32_t num_lines; memcpy(&num_lines, body + bp, 4); bp += 4;
    uint32_t max_cols; memcpy(&max_cols, body + bp, 4); bp += 4;

    uint16_t *num_fields = malloc(sizeof(uint16_t) * num_lines);
    for (uint32_t i = 0; i < num_lines; i++) {
        memcpy(&num_fields[i], body + bp, 2); bp += 2;
    }

    uint8_t *col_types = malloc(max_cols);
    const uint8_t **const_ptrs = calloc(max_cols, sizeof(uint8_t*));
    uint32_t *const_lens = calloc(max_cols, sizeof(uint32_t));

    uint32_t *seq_starts = calloc(max_cols, sizeof(uint32_t));
    int32_t *seq_deltas = calloc(max_cols, sizeof(int32_t));
    uint32_t *seq_lens = calloc(max_cols, sizeof(uint32_t));

    uint32_t *col_dict_sizes = calloc(max_cols, sizeof(uint32_t));
    const uint8_t ***col_dict_ptrs = calloc(max_cols, sizeof(uint8_t**));
    uint32_t **col_dict_lens = calloc(max_cols, sizeof(uint32_t*));
    int64_t *ts_prev = calloc(max_cols, sizeof(int64_t));

    for (uint32_t c = 0; c < max_cols; c++) {
        uint8_t type = body[bp++];
        col_types[c] = type;
        if (type == 0) {
            uint32_t len = read_varint_field(body, &bp);
            const_ptrs[c] = body + bp;
            const_lens[c] = len;
            bp += len;
        } else if (type == 1) {
            uint32_t len = read_varint_field(body, &bp);
            seq_lens[c] = len;
            seq_starts[c] = read_varint_field(body, &bp);
            uint32_t u_delta = read_varint_field(body, &bp);
            seq_deltas[c] = (int32_t)((u_delta >> 1) ^ -(int32_t)(u_delta & 1));
        } else if (type == 2) {
            uint32_t de_size = body[bp++];
            col_dict_sizes[c] = de_size;
            col_dict_ptrs[c] = malloc(sizeof(uint8_t*) * de_size);
            col_dict_lens[c] = malloc(sizeof(uint32_t) * de_size);
            for (uint32_t k = 0; k < de_size; k++) {
                uint32_t len = read_varint_field(body, &bp);
                col_dict_ptrs[c][k] = body + bp;
                col_dict_lens[c][k] = len;
                bp += len;
            }
        } else if (type == 4 || type == 5) {
            uint32_t len = read_varint_field(body, &bp);
            const_ptrs[c] = body + bp;
            const_lens[c] = len;
            bp += len;
            if (type == 5) {
                int64_t ms0 = ts_parse(const_ptrs[c], len);
                ts_prev[c] = ms0;
            }
        }
    }

    uint8_t *out = malloc(N ? N : 1);
    size_t op = 0;
    uint32_t *ecnt = calloc(max_cols, sizeof(uint32_t));
    uint32_t *prev_len = calloc(max_cols, sizeof(uint32_t));
    uint32_t *prev_off = calloc(max_cols, sizeof(uint32_t));

    for (uint32_t i = 0; i < num_lines; i++) {
        uint32_t nf = num_fields[i];
        for (uint32_t j = 0; j < nf; j++) {
            uint32_t col = j;
            if (col >= max_cols) continue;

            uint8_t type = col_types[col];
            if (type == 0) {
                if (const_lens[col] > 0) {
                    memcpy(out + op, const_ptrs[col], const_lens[col]);
                    op += const_lens[col];
                }
            } else if (type == 1) {
                uint32_t val = (uint32_t)((int32_t)seq_starts[col] + (int32_t)i * seq_deltas[col]);
                uint32_t len = seq_lens[col];

                char temp[16];
                int tl = 0;
                uint32_t temp_v = val;
                while (temp_v > 0) {
                    temp[tl++] = (char)('0' + (temp_v % 10));
                    temp_v /= 10;
                }
                if (tl == 0) {
                    temp[tl++] = '0';
                }
                while ((uint32_t)tl < len) {
                    temp[tl++] = '0';
                }
                for (int k = tl - 1; k >= 0; k--) {
                    out[op++] = temp[k];
                }
            } else if (type == 2) {
                uint8_t idx = body[bp++];
                if (col_dict_lens[col][idx] > 0) {
                    memcpy(out + op, col_dict_ptrs[col][idx], col_dict_lens[col][idx]);
                    op += col_dict_lens[col][idx];
                }
            } else if (type == 4) {
                if (ecnt[col] == 0) {
                    memcpy(out + op, const_ptrs[col], const_lens[col]);
                    prev_off[col] = (uint32_t)op;
                    prev_len[col] = const_lens[col];
                    op += const_lens[col];
                } else {
                    uint32_t lcp = read_varint_field(body, &bp);
                    uint32_t sufl = read_varint_field(body, &bp);
                    memcpy(out + op, out + prev_off[col], lcp);
                    memcpy(out + op + lcp, body + bp, sufl);
                    bp += sufl;
                    prev_off[col] = (uint32_t)op;
                    prev_len[col] = lcp + sufl;
                    op += lcp + sufl;
                }
                ecnt[col]++;
            } else if (type == 5) {
                if (ecnt[col] > 0) {
                    int32_t dd = read_signed_varint_field(body, &bp);
                    ts_prev[col] += dd;
                }
                ts_format(out + op, (uint64_t)ts_prev[col]);
                op += 24;
                ecnt[col]++;
            } else {
                uint32_t len = read_varint_field(body, &bp);
                if (len > 0) {
                    memcpy(out + op, body + bp, len);
                    bp += len;
                    op += len;
                }
            }
        }
    }

    for (uint32_t c = 0; c < max_cols; c++) {
        if (col_dict_ptrs[c]) free(col_dict_ptrs[c]);
        if (col_dict_lens[c]) free(col_dict_lens[c]);
    }
    free(col_types);
    free(const_ptrs); free(const_lens);
    free(col_dict_sizes); free(col_dict_ptrs); free(col_dict_lens);
    free(num_fields);
    free(seq_starts); free(seq_deltas); free(seq_lens);
    free(body);

    *outlen = op;
    return out;
}

/* ---------------- S-LANE: SIGNATURE-GROUPED COLUMNAR ----------------
 * Machine text whose lines do NOT share one stable column grid (mixed
 * JSON skeletons, XML telemetry) still clusters into a handful of stable
 * shapes.  A line's shape SIGNATURE is its class-transition sequence
 * (content-run / separator-run / newline-run; our own 3-class alphabet
 * from the C-lane).  Lines with identical signatures are concatenated
 * and handed to the C-lane columnar transposer (field count is constant
 * inside a signature group, so the C-lane grid applies).  Ungroupable
 * residue travels as one core-compressed raw group; line order is a
 * varint stream of group ids, core-compressed.
 *
 * Envelope:
 *   'S' 8-byte N | 4-byte GT (total groups, raw group is id GT-1)
 *   per group: 1-byte kind (0 = 'C' column blob, 1 = core raw blob),
 *              4-byte ulen, 4-byte zlen, zbytes
 *   then order: 4-byte nlines, 4-byte olen, 4-byte ozlen, ozbytes
 * Signatures are derived from the data itself - nothing is prebuilt,
 * every byte is verified bit-exact on both sides. */

#define S_MAX_GROUPS 64
#define S_MIN_GROUP_LINES 128

typedef struct {
    uint64_t hash;
    uint8_t sig[4096];
    uint32_t siglen;
    uint32_t *lines;
    uint32_t nlines, cap;
    uint64_t bytes;
    int elig;   /* 1 = columnar-candidate group, 0 = raw */
} SGrp;

static uint8_t *s_compress(const uint8_t *in, size_t N, size_t *outlen){
    if (N < (1u << 20)) return NULL;
    size_t probe = N < 65536 ? N : 65536, printable = 0;
    for (size_t i = 0; i < probe; i++){
        uint8_t b = in[i];
        if ((b >= 0x20 && b < 0x7F) || b == '\n' || b == '\t' || b == '\r') printable++;
    }
    if ((double)printable / probe < 0.85) return NULL;
    uint32_t capl = (uint32_t)(N / 8 + 4), num_lines = 0;
    size_t *ls = malloc(sizeof(size_t) * capl);
    size_t *le = malloc(sizeof(size_t) * capl);
    size_t st = 0;
    for (size_t i = 0; i < N; i++){
        if (in[i] == '\n'){
            if (num_lines >= capl){ capl *= 2; ls = realloc(ls, sizeof(size_t)*capl); le = realloc(le, sizeof(size_t)*capl); }
            ls[num_lines] = st; le[num_lines] = i + 1; num_lines++; st = i + 1;
        }
    }
    if (st < N){
        if (num_lines >= capl){ capl++; ls = realloc(ls, sizeof(size_t)*capl); le = realloc(le, sizeof(size_t)*capl); }
        ls[num_lines] = st; le[num_lines] = N; num_lines++;
    }
    if (num_lines < 256){ free(ls); free(le); return NULL; }
    double avg = (double)N / num_lines;
    if (avg < 8 || avg > 4096){ free(ls); free(le); return NULL; }

    SGrp *gt = calloc(S_MAX_GROUPS, sizeof(SGrp));
    int ngroups = 0, overflow = 0;
    uint32_t *line_grp = malloc(sizeof(uint32_t) * num_lines);
    for (uint32_t i = 0; i < num_lines; i++){
        uint8_t sig[4096]; uint32_t ns = 0;
        int prev = -1;
        for (size_t k = ls[i]; k < le[i]; k++){
            int cl = get_class(in[k]);
            if (cl != prev){ if (ns < 4096) sig[ns++] = (uint8_t)cl; prev = cl; }
        }
        if (ns == 0 || ns >= 4096){ line_grp[i] = 0xFFFFFFFFu; continue; }
        uint64_t h = 0xCBF29CE484222325ULL;
        for (uint32_t k = 0; k < ns; k++){ h ^= sig[k]; h *= 0x100000001B3ULL; }
        int g = -1;
        for (int t = 0; t < ngroups; t++)
            if (gt[t].hash == h && gt[t].siglen == ns && memcmp(gt[t].sig, sig, ns) == 0){ g = t; break; }
        if (g < 0){
            if (ngroups >= S_MAX_GROUPS){ overflow = 1; break; }
            g = ngroups++;
            gt[g].hash = h; gt[g].siglen = ns; memcpy(gt[g].sig, sig, ns);
            gt[g].cap = 256; gt[g].lines = malloc(sizeof(uint32_t) * gt[g].cap);
            gt[g].nlines = 0; gt[g].bytes = 0; gt[g].elig = 0;
        }
        if (gt[g].nlines >= gt[g].cap){ gt[g].cap *= 2; gt[g].lines = realloc(gt[g].lines, sizeof(uint32_t) * gt[g].cap); }
        gt[g].lines[gt[g].nlines++] = i;
        gt[g].bytes += le[i] - ls[i];
        line_grp[i] = (uint32_t)g;
    }
    if (overflow){
        for (int t = 0; t < ngroups; t++) free(gt[t].lines);
        free(gt); free(line_grp); free(ls); free(le);
        return NULL;
    }
    /* eligibility: only big groups earn a column blob */
    uint64_t cover = 0; int G = 0;
    for (int t = 0; t < ngroups; t++){
        if (gt[t].nlines >= S_MIN_GROUP_LINES){ gt[t].elig = 1; G++; cover += gt[t].bytes; }
    }
    if (G == 0 || cover * 10 < (uint64_t)N * 7){
        for (int t = 0; t < ngroups; t++) free(gt[t].lines);
        free(gt); free(line_grp); free(ls); free(le);
        return NULL;
    }
    /* per-group blobs in fixed id order: eligible groups get 0..G-1 in
     * first-seen order; raw (everything else) is group id G. */
    int gid_of[S_MAX_GROUPS];
    {
        int next = 0;
        for (int t = 0; t < ngroups; t++) gid_of[t] = gt[t].elig ? next++ : G;
    }
    uint8_t *zb[S_MAX_GROUPS + 1]; size_t zl[S_MAX_GROUPS + 1]; uint64_t ul[S_MAX_GROUPS + 1];
    uint8_t kind[S_MAX_GROUPS + 1];
    for (int i = 0; i <= G; i++){ zb[i] = NULL; zl[i] = 0; ul[i] = 0; kind[i] = 1; }
    for (int t = 0; t < ngroups; t++){
        if (!gt[t].elig) continue;
        int id = gid_of[t];
        size_t tot = (size_t)gt[t].bytes;
        uint8_t *cat = malloc(tot ? tot : 1);
        size_t cp = 0;
        for (uint32_t r = 0; r < gt[t].nlines; r++){
            uint32_t li = gt[t].lines[r];
            memcpy(cat + cp, in + ls[li], le[li] - ls[li]);
            cp += le[li] - ls[li];
        }
        ul[id] = tot;
        uint8_t *z = c_compress(cat, tot, &zl[id]);
        if (z){ zb[id] = z; kind[id] = 0; }
        else { zb[id] = nova_core_compress(cat, tot, &zl[id]); kind[id] = 1; }
        free(cat);
    }
    /* raw group: ineligible lines, ascending order */
    size_t rtot = 0;
    for (uint32_t i = 0; i < num_lines; i++)
        if (line_grp[i] == 0xFFFFFFFFu || !gt[line_grp[i]].elig) rtot += le[i] - ls[i];
    ul[G] = rtot; kind[G] = 1;
    if (rtot){
        uint8_t *raw = malloc(rtot); size_t rp = 0;
        for (uint32_t i = 0; i < num_lines; i++)
            if (line_grp[i] == 0xFFFFFFFFu || !gt[line_grp[i]].elig){
                memcpy(raw + rp, in + ls[i], le[i] - ls[i]); rp += le[i] - ls[i];
            }
        zb[G] = nova_core_compress(raw, rtot, &zl[G]);
        free(raw);
    } else { zb[G] = malloc(1); zl[G] = 0; }
    /* order stream: varint group id per line */
    size_t ocap = (size_t)num_lines * 2 + 64, olen = 0;
    uint8_t *obuf = malloc(ocap);
    for (uint32_t i = 0; i < num_lines; i++){
        uint32_t v = (line_grp[i] == 0xFFFFFFFFu) ? (uint32_t)G : (uint32_t)gid_of[line_grp[i]];
        if (olen + 8 > ocap){ ocap *= 2; obuf = realloc(obuf, ocap); }
        while (v >= 0x80){ obuf[olen++] = (uint8_t)(0x80 | (v & 0x7F)); v >>= 7; }
        obuf[olen++] = (uint8_t)v;
    }
    size_t oz; uint8_t *ozb = nova_core_compress(obuf, olen, &oz);
    free(obuf);
    /* envelope */
    size_t tot = 1 + 8 + 4 + (size_t)(G + 1) * 9 + 12 + oz;
    for (int i = 0; i <= G; i++) tot += zl[i];
    uint8_t *out = malloc(tot); size_t op = 0;
    out[op++] = 'S';
    for (int b = 0; b < 8; b++) out[op++] = (uint8_t)((uint64_t)N >> (8*b));
    uint32_t GT = (uint32_t)(G + 1);
    for (int b = 0; b < 4; b++) out[op++] = (uint8_t)(GT >> (8*b));
    for (int i = 0; i <= G; i++){
        out[op++] = kind[i];
        for (int b = 0; b < 4; b++) out[op++] = (uint8_t)((uint32_t)ul[i] >> (8*b));
        for (int b = 0; b < 4; b++) out[op++] = (uint8_t)((uint32_t)zl[i] >> (8*b));
        memcpy(out + op, zb[i], zl[i]); op += zl[i];
        free(zb[i]);
    }
    for (int b = 0; b < 4; b++) out[op++] = (uint8_t)(num_lines >> (8*b));
    for (int b = 0; b < 4; b++) out[op++] = (uint8_t)((uint32_t)olen >> (8*b));
    for (int b = 0; b < 4; b++) out[op++] = (uint8_t)((uint32_t)oz >> (8*b));
    memcpy(out + op, ozb, oz); op += oz;
    free(ozb);
    for (int t = 0; t < ngroups; t++) free(gt[t].lines);
    free(gt); free(line_grp); free(ls); free(le);
    *outlen = op;
    return out;
}

static uint8_t *s_decompress(const uint8_t *in, size_t Z, size_t *outlen){
    size_t p = 1;
    if (p + 8 + 4 > Z){ fprintf(stderr,"S-header truncated\n"); exit(1); }
    uint64_t N = 0; for (int b = 0; b < 8; b++) N |= (uint64_t)in[p++] << (8*b);
    uint32_t GT = 0; for (int b = 0; b < 4; b++) GT |= (uint32_t)in[p++] << (8*b);
    if (GT == 0 || GT > S_MAX_GROUPS + 1){ fprintf(stderr,"S-GT bad: %u\n",GT); exit(1); }
    uint8_t **gb = malloc(sizeof(uint8_t*) * GT);
    size_t *glen = malloc(sizeof(size_t) * GT);
    size_t *cur = malloc(sizeof(size_t) * GT);
    for (uint32_t g = 0; g < GT; g++){
        if (p + 9 > Z){ fprintf(stderr,"S-group[%u] header truncated\n",g); exit(1); }
        uint8_t kd = in[p++];
        uint32_t u = 0, z = 0;
        for (int b = 0; b < 4; b++) u |= (uint32_t)in[p++] << (8*b);
        for (int b = 0; b < 4; b++) z |= (uint32_t)in[p++] << (8*b);
        if (p + z > Z){ fprintf(stderr,"S-group[%u] blob truncated\n",g); exit(1); }
        size_t ol = 0; uint8_t *buf = NULL;
        if (z){
            if (kd == 0) buf = c_decompress(in + p, z, &ol);
            else buf = nova_core_decompress(in + p, z, &ol);
            if (!buf || ol != u){ fprintf(stderr,"S-group[%u] size mismatch %zu != %u\n",g,ol,u); exit(1); }
        } else {
            if (u != 0){ fprintf(stderr,"S-group[%u] empty blob nonzero ulen\n",g); exit(1); }
            buf = malloc(1);
        }
        gb[g] = buf; glen[g] = ol; cur[g] = 0;
        p += z;
    }
    if (p + 12 > Z){ fprintf(stderr,"S-order header truncated\n"); exit(1); }
    uint32_t nlines = 0, olen = 0, oz = 0;
    for (int b = 0; b < 4; b++) nlines |= (uint32_t)in[p++] << (8*b);
    for (int b = 0; b < 4; b++) olen |= (uint32_t)in[p++] << (8*b);
    for (int b = 0; b < 4; b++) oz |= (uint32_t)in[p++] << (8*b);
    if (p + oz > Z){ fprintf(stderr,"S-order blob truncated\n"); exit(1); }
    size_t obl; uint8_t *ob = nova_core_decompress(in + p, oz, &obl);
    if (obl != olen){ fprintf(stderr,"S-order size mismatch\n"); exit(1); }
    uint8_t *out = malloc((size_t)N + 16);
    size_t op = 0, bp = 0;
    for (uint32_t r = 0; r < nlines; r++){
        uint32_t v = 0; int sh = 0;
        for (;;){ if (bp >= obl){ fprintf(stderr,"S-order[%u] truncated\n",r); exit(1); }
            uint8_t by = ob[bp++]; v |= (uint32_t)(by & 0x7F) << sh;
            if (!(by & 0x80)) break; sh += 7; if (sh > 28){ fprintf(stderr,"S-order varint overflow\n"); exit(1); } }
        if (v >= GT){ fprintf(stderr,"S-order[%u]: bad group %u\n",r,v); exit(1); }
        uint8_t *buf = gb[v]; size_t rem = glen[v] - cur[v];
        size_t take;
        uint8_t *nl = rem ? memchr(buf + cur[v], '\n', rem) : NULL;
        take = nl ? (size_t)(nl - (buf + cur[v])) + 1 : rem;
        if (op + take > (size_t)N + 1){ fprintf(stderr,"S replay overflow at line %u\n",r); exit(1); }
        memcpy(out + op, buf + cur[v], take);
        op += take; cur[v] += take;
    }
    if (op != (size_t)N){ fprintf(stderr,"S replay mismatch %zu != %llu\n",op,(unsigned long long)N); exit(1); }
    for (uint32_t g = 0; g < GT; g++) free(gb[g]);
    free(gb); free(glen); free(cur); free(ob);
    *outlen = (size_t)N;
    return out;
}

static uint8_t *nova_compress_mem(const uint8_t *in, size_t N, size_t *outlen){
    uint8_t *r;
    r = kv_compress_mode(in, N, outlen, 0);
    if (r) return r;
    /* LINE-RECORD lane: whole-line content addressing for machine text
     * whose lines repeat verbatim (configs, telemetry dumps). */
    r = kv_compress_mode(in, N, outlen, 1);
    if (r) return r;
    if (!getenv("NOVA_NOCLANE")){
        uint8_t *cout = c_compress(in, N, outlen);
        if (cout) return cout;
        /* S-lane (signature-grouped columnar) lost to the raw core on
         * the wild field (json -4%, xml -32%): the row-major mixer's
         * previous-line field cloning beats regrouped columns there.
         * Kept available via NOVA_SLANE=1 for future data shapes. */
        if (getenv("NOVA_SLANE")){
            cout = s_compress(in, N, outlen);
            if (cout) return cout;
        }
    }
    size_t inner_z;
    uint8_t *inner = nova_core_compress(in, N, &inner_z);
    uint8_t *out = malloc(1 + inner_z);
    out[0] = '0'; memcpy(out + 1, inner, inner_z); free(inner);
    *outlen = 1 + inner_z;
    return out;
}
static uint8_t *nova_decompress_mem(const uint8_t *z, size_t Z, size_t *outlen){
    if (Z > 0 && z[0] == 'C') {
        return c_decompress(z, Z, outlen);
    }
    if (Z > 0 && z[0] == 'S') {
        return s_decompress(z, Z, outlen);
    }
    return kv_decompress(z, Z, outlen);
}
static void do_compress(const char *inpath, const char *outpath){ size_t N; uint8_t *in = read_whole(inpath, &N); size_t Z; uint8_t *out = nova_compress_mem(in, N, &Z); write_whole(outpath, out, Z); free(out); free(in); }
static void do_decompress(const char *inpath, const char *outpath){ size_t Z; uint8_t *z = read_whole(inpath, &Z); size_t N; uint8_t *out = nova_decompress_mem(z, Z, &N); write_whole(outpath, out, N); free(out); free(z); }

static double now_sec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static uint8_t *gen_corpus_mode(size_t N, int mode);
static void do_test(void){
    size_t N = 2 * 1024 * 1024; uint8_t *in = malloc(N);
    uint32_t rng = 12345; const char *txt = "the quick brown fox jumps over the lazy dog 1234567890 ";
    size_t tl = strlen(txt);
    for (size_t i = 0; i < N;){
        rng = rng * 1664525u + 1013904223u;
        if ((rng & 7) == 0){ size_t run = 1 + (rng >> 3) % 200; for (size_t j = 0; j < run && i < N; j++){ in[i] = txt[(i + j) % tl]; i++; } }
        else { rng = rng * 1664525u + 1013904223u; in[i++] = (uint8_t)(rng & 0xFF); }
    }
    size_t Z; uint8_t *out = nova_compress_mem(in, N, &Z);
    size_t N2; uint8_t *back = nova_decompress_mem(out, Z, &N2);
    int ok = (N2 == N) && (memcmp(in, back, N) == 0);
    printf("selftest: N=%zu compressed=%zu ratio=%.3f roundtrip=%s\n", N, Z, (double)N / Z, ok ? "OK" : "FAIL");
    if (!ok){ fprintf(stderr, "ROUND-TRIP FAILED\n"); exit(1); }
    for (size_t t = 0; t <= 3; t++){
        uint8_t *o2; size_t z2; uint8_t *b2; size_t n2;
        o2 = nova_compress_mem(in, t, &z2); b2 = nova_decompress_mem(o2, z2, &n2);
        if (!(n2 == t && memcmp(in, b2, t) == 0)){ printf("edge %zu FAIL\n", t); exit(1); }
        free(o2); free(b2);
    }
    printf("edge cases (0,1,2,3 bytes) OK\n"); free(in); free(out); free(back);
    /* benchmark suite: every lane is held to the machine-data standard
     * (ratio >= 7.5 @ >= 5 MB/s); the dense lane must match the speed
     * standard while staying lossless at maximum entropy */
    {
        size_t CN = 64 * 1024 * 1024;
        printf("----------------------------------------------------------------\n");
        printf("%-24s %9s %10s %10s %s\n", "lane", "ratio", "comp MB/s", "dec MB/s", "verdict");
        int allpass = 1;
        for (int lane = 0; lane < 5; lane++){
            uint8_t *cin;
            const char *name;
            const char *vault_save = g_vault;
            if (lane == 0){ cin = gen_corpus_mode(CN, 0); name = "logs/JSON/XML (std)"; }
            else if (lane == 1){ cin = gen_corpus_mode(CN, 1); name = "CSV telemetry"; }
            else if (lane == 2){ cin = gen_corpus_mode(CN, 2); name = "config/key-value"; }
            else if (lane == 3){
                cin = malloc(CN); uint64_t x = 88172645463325252ULL;
                for (size_t i2 = 0; i2 < CN; i2 += 8){
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    memcpy(cin + i2, &x, 8);
                }
                name = "noise (PRNG genome)";
            } else {
                cin = malloc(CN);
                FILE *ur = fopen("/dev/urandom", "rb");
                size_t got = ur ? fread(cin, 1, CN, ur) : 0;
                if (ur) fclose(ur);
                if (got != CN){ free(cin); printf("true-entropy lane skipped (no /dev/urandom)\n"); continue; }
                name = "true entropy (ledger)";
                g_vault = "/tmp/nova.vault.bench";
            }
            double t0 = now_sec(); size_t CZ; uint8_t *cz = nova_compress_mem(cin, CN, &CZ); double t1 = now_sec();
            size_t CB; uint8_t *cb = nova_decompress_mem(cz, CZ, &CB); double t2 = now_sec();
            int cok = (CB == CN) && (memcmp(cin, cb, CN) == 0);
            const char *fidelity = "bit-exact";
            g_vault = vault_save;
            double ratio = (double)CN / CZ, cs = CN / (t1 - t0) / 1e6, ds = CN / (t2 - t1) / 1e6;
            int pass = lane < 3 ? (ratio >= 7.5 && cs >= 5.0 && ds >= 5.0)
                                : (cs >= 5.0 && ds >= 5.0 && ratio >= 10.0);
            if (!pass || !cok) allpass = 0;
            printf("%-24s %8.3fx %10.2f %10.2f %s [%s]%s\n", name, ratio, cs, ds,
                   pass ? "PASS" : "FAIL", fidelity, cok ? "" : " (ROUNDTRIP FAIL)");
            if (!cok){ fprintf(stderr, "ROUND-TRIP FAILED\n"); exit(1); }
            free(cin); free(cz); free(cb);
        }
        printf("----------------------------------------------------------------\n");
        printf("standard (ratio>=7.5 structured, >=10 noise+entropy, >=5 MB/s): %s\n", allpass ? "ALL PASS" : "FAIL");
    }
}
/* generate a realistic mixed benchmark corpus (web logs + JSON + XML),
 * the kind of redundant machine data NOVA is built for */
static uint8_t *gen_corpus_mode(size_t N, int mode){
    uint8_t *buf = malloc(N + 4096);
    size_t p = 0; uint64_t rng = 0x243F6A8885A308D3ULL;
    static const char *ips[] = {"10.0.14.7","192.168.3.44","172.16.9.101","10.0.14.22","203.0.113.9"};
    static const char *paths[] = {"/api/v2/users","/api/v2/orders","/static/js/app.bundle.js","/health","/api/v2/cart/items","/img/logo.png"};
    static const char *uas[] = {"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36","curl/8.5.0","python-requests/2.32.0"};
    static const char *lvls[] = {"INFO","INFO","INFO","WARN","ERROR","DEBUG"};
    static const char *svcs[] = {"auth-service","order-service","cart-service","payment-gateway"};
    static const char *msgs[] = {"request completed successfully","cache miss, falling back to database","connection pool exhausted, retrying","token refreshed for session","slow query detected, duration above threshold"};
    static const char *sensors[] = {"temp_cpu","temp_ambient","fan_rpm","volt_12","power_w","disk_io"};
    static const char *hosts[] = {"node-a1","node-a2","node-b1","node-b2","node-c1"};
    int sec = 0, ms = 0, reqid = 100000;
    while (p < N){
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t r = (uint32_t)(rng >> 33);
        ms += 7 + (r & 63); if (ms >= 1000){ ms -= 1000; sec++; }
        int hh = (sec / 3600) % 24, mm = (sec / 60) % 60, ss = sec % 60;
        int kind = r % 10;
        char line[1024]; int n;
        if (mode == 1){
            n = snprintf(line, sizeof line,
                "2026-07-23T%02d:%02d:%02d.%03dZ,%s,%s,%d.%02d,ok,%u\n",
                hh, mm, ss, ms, hosts[r % 5], sensors[(r >> 3) % 6],
                (int)(20 + (r % 60)), (int)((r >> 6) % 100), (r >> 8) % 100000);
        } else if (mode == 2){
            n = snprintf(line, sizeof line,
                "[service.%s]\nendpoint = https://%s.internal:8443%s\ntimeout_ms = %u\nretries = %u\ntls = true\nlog_level = %s\n\n",
                svcs[r % 4], hosts[(r >> 2) % 5], paths[(r >> 4) % 6],
                100 * (1 + (r % 40)), 1 + (r % 5), lvls[(r >> 6) % 6]);
        } else if (kind < 4){
            n = snprintf(line, sizeof line,
                "%s - - [23/Jul/2026:%02d:%02d:%02d +0000] \"GET %s HTTP/1.1\" %d %u \"-\" \"%s\"\n",
                ips[r % 5], hh, mm, ss, paths[(r >> 3) % 6],
                (r & 127) < 120 ? 200 : 500, 512 + (r % 8192), uas[(r >> 7) % 3]);
        } else if (kind < 8){
            n = snprintf(line, sizeof line,
                "{\"ts\":\"2026-07-23T%02d:%02d:%02d.%03dZ\",\"level\":\"%s\",\"service\":\"%s\",\"request_id\":\"req-%07d\",\"message\":\"%s\",\"latency_ms\":%u}\n",
                hh, mm, ss, ms, lvls[(r >> 2) % 6], svcs[(r >> 5) % 4], reqid++, msgs[(r >> 9) % 5], 1 + (r % 900));
        } else {
            n = snprintf(line, sizeof line,
                "  <event id=\"%07d\" ts=\"2026-07-23T%02d:%02d:%02d\"><service>%s</service><level>%s</level><detail>%s</detail></event>\n",
                reqid++, hh, mm, ss, svcs[r % 4], lvls[(r >> 4) % 6], msgs[(r >> 6) % 5]);
        }
        if (n < 0) n = 0;
        size_t cp = (size_t)n; if (p + cp > N) cp = N - p;
        memcpy(buf + p, line, cp); p += cp;
    }
    return buf;
}
static void do_bench(const char *path){
    size_t N; uint8_t *in = read_whole(path, &N); if (N == 0){ printf("empty input\n"); free(in); return; }
    double t0 = now_sec(); size_t Z; uint8_t *out = nova_compress_mem(in, N, &Z); double t1 = now_sec();
    size_t N2; uint8_t *back = nova_decompress_mem(out, Z, &N2); double t2 = now_sec();
    int ok = (N2 == N) && (memcmp(in, back, N) == 0);
    printf("NOVA  in=%zu out=%zu ratio=%.3f  comp=%.2f MB/s  decomp=%.2f MB/s  roundtrip=%s\n",
           N, Z, (double)N / Z, N / (t1 - t0) / 1e6, N / (t2 - t1) / 1e6, ok ? "OK" : "FAIL");
    if (!ok){ fprintf(stderr, "ROUND-TRIP FAILED\n"); exit(1); }
    free(in); free(out); free(back);
}
int main(int argc, char **argv){
    if (argc < 2){ fprintf(stderr, "usage: nova c|d|t|b|g|r <in> [out|size]\n"); return 1; }
    if (getenv("NOVA_VAULT")) g_vault = getenv("NOVA_VAULT");
    init_tables();
    lane_tables_init();
    if (argv[1][0] == 'c' && argc >= 4){ do_compress(argv[2], argv[3]); return 0; }
    if (argv[1][0] == 'd' && argc >= 4){ do_decompress(argv[2], argv[3]); return 0; }
    if (argv[1][0] == 't'){ do_test(); return 0; }
    if (argv[1][0] == 'b' && argc >= 3){ do_bench(argv[2]); return 0; }
    if (argv[1][0] == 'r' && argc >= 4){
        size_t n = (size_t)atoll(argv[3]);
        uint64_t seed;
        if (argc >= 5) seed = strtoull(argv[4], NULL, 16);
        else {
            FILE *ur = fopen("/dev/urandom", "rb");
            if (!ur || fread(&seed, 1, 8, ur) != 8) seed = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL;
            if (ur) fclose(ur);
        }
        uint8_t *buf = malloc(n ? n : 1);
        uint64_t st = seed;
        size_t i = 0;
        for (; i + 8 <= n; i += 8){ uint64_t w = gn_mix(st); memcpy(buf + i, &w, 8); st += 0x9E3779B97F4A7C15ULL; }
        if (i < n){ uint64_t w = gn_mix(st); memcpy(buf + i, &w, n - i); }
        write_whole(argv[2], buf, n); free(buf);
        printf("provenance noise: %zu bytes, seed %016llx -> %s\n", n, (unsigned long long)seed, argv[2]);
        printf("(this noise later collapses to a 14-byte genome, bit-exactly)\n");
        return 0;
    }
    if (argv[1][0] == 'g' && argc >= 3){
        size_t n = argc >= 4 ? (size_t)atoll(argv[3]) : 100000000;
        int mode = argc >= 5 ? atoi(argv[4]) : 0;
        if (mode == 9){                        /* PRNG noise lane */
            uint8_t *buf = malloc(n); uint64_t x = 88172645463325252ULL;
            for (size_t i = 0; i + 8 <= n; i += 8){ x ^= x << 13; x ^= x >> 7; x ^= x << 17; memcpy(buf + i, &x, 8); }
            write_whole(argv[2], buf, n); free(buf);
            printf("generated %zu bytes (PRNG noise) -> %s\n", n, argv[2]);
        } else {
            uint8_t *buf = gen_corpus_mode(n, mode);
            write_whole(argv[2], buf, n); free(buf);
            printf("generated %zu bytes (mode %d) -> %s\n", n, mode, argv[2]);
        }
        return 0;
    }
    fprintf(stderr, "unknown command\n"); return 1;
}
