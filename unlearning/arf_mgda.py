"""
arf_mgda.py
===========
Depends on prune_unlearn_5methods.py's shared machinery (PConfig, setup(),
build_model(), WDataset, prune_by_ratio(), full_report(), _fresh_pruned_model(),
writer_acc(), rec_quality(), apply_masks(), _infinite()) being importable or
already defined in the same execution context -- paste this AFTER
prune_unlearn_5methods.py, same convention as retrain_preprocessing_override.py.

Two experiments, both adapting Wang, Zhang, Tian & Yu, "Machine Unlearning via
Representation Forgetting With Parameter Self-Sharing" (IEEE TIFS, submitted),
to a deterministic ViT (no VIB architecture, no stochastic latent to reweight):

  EXPERIMENT A -- Adversarial Representation Forgetting (ARF). A domain-
  adversarial (gradient-reversal) probe over the ViT CLS embedding, continually
  retrained rather than a fixed target the encoder learns to fool once. Result
  (this project's own audit): does not outperform output-level methods on the
  frozen-encoder probe or relearning attack -- reported as evidence of
  representational redundancy, not implementation failure. See
  exp6_arf_ablation.py for the adversary-capacity/cadence/loss-weight ablation
  that stress-tests this conclusion.

  EXPERIMENT B -- MGDA-style dynamic gradient weighting (RFU-SS's Eq. 11-14,
  architecture-agnostic). Applied only to WIC and Random Labeling, which
  already combine a per-step forget loss and a per-step retain loss; Fisher/
  Amnesiac/DELETE are two-phase shock-then-repair methods with no per-step
  joint objective to reweight and are deliberately excluded. Result: Random
  Labeling+MGDA is statistically indistinguishable from fixed-weight Random
  Labeling; WIC+MGDA's apparent probe-gap win does not survive a retain-writer
  collapse check (Figure fig:targeted-vs-collapse) -- general representational
  damage, not targeted forgetting.

call main_arf_mgda() to run both experiments end-to-end (fresh Stage I prune,
then ARF, then WIC-MGDA, then RL-MGDA, each 800 iterations by default).
"""

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A -- ADVERSARIAL REPRESENTATION FORGETTING (ARF)
#
# Adapted from Wang, Zhang, Tian & Yu, "Machine Unlearning via Representation
# Forgetting With Parameter Self-Sharing" (IEEE TIFS, submitted), WITHOUT their
# Information Bottleneck architecture: this model's ViT CLS embedding is a
# deterministic vector, not a stochastic VIB latent z ~ p(z|x), so there is no
# I(Z;X) to estimate and no KLD-based RFU loss to port literally. What IS
# ported is the underlying diagnosis both that paper and this project's own
# audit converge on: output-level objectives (WIC's KL-to-uniform, Random
# Labeling's relabelling, Amnesiac's gradient ascent, DELETE's masked
# distillation) only ever touch the classification head sitting on top of the
# representation -- per this project's own audit, a FRESH probe trained after
# the fact on the frozen encoder still recovers 63-99% of forget-writer
# identity regardless of which of those four methods was used.
#
# ARF instead attacks the CLS representation directly, via a domain-
# adversarial (gradient-reversal) probe that is CONTINUALLY retrained on the
# current embedding at every step -- not a fixed target the encoder learns to
# fool once. This is the mechanism most faithful to RFU-SS's actual claim
# (representation-level mutual-information suppression) that is buildable
# without retraining the base model into a VIB architecture from scratch.
#
# The model's OWN writer_head is ALSO pushed toward uniform on the forget
# batch (identical mechanism to WIC) purely so "forgotten-writer output
# performance" (metric 1) stays comparable across all six methods. The actual
# test of this method is metric 3 (an independent, freshly-trained-from-
# scratch post-hoc probe, run later by the audit kernel) and metric 4 (the
# retain-replay-controlled relearning attack) -- neither of which ever sees
# this training-time adversarial probe.
# ══════════════════════════════════════════════════════════════════════════════
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class AdversarialProbe(nn.Module):
    """Fresh, continually-retrained adversary over the ViT CLS embedding,
    predicting WHICH of the 20 forget writers an embedding belongs to."""

    def __init__(self, d_model, n_forget):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, n_forget),
        )

    def forward(self, z):
        return self.net(z)

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class ARFConfig:
    LAMBDA_ADV = 1.0     # weight on the GRL-adversarial representation loss
    LAMBDA_HEAD = 1.0    # weight on suppressing the model's own writer_head (keeps metric 1 comparable to WIC/RL/Amnesiac/DELETE)
    LAMBDA_RETAIN = 1.0
    LAMBDA_REC = 0.5
    GRL_LAMBDA = 1.0     # gradient-reversal strength
    PROBE_RESET_EVERY = 100   # periodic reinit so the adversary can't go stale / get stuck in a weak local optimum


def adversarial_representation_forget(model, ctx, masks, iters, checkpoint_steps, tag,
                                       cfg=PConfig, acfg=ARFConfig):
    vocab, fe = ctx["vocab"], ctx["fe"]
    forget_df, retain_df = ctx["forget_df"], ctx["retain_df"]
    n_classes = ctx["n_classes"]

    forget_writers = sorted(int(i) for i in forget_df["wid"].unique())
    n_forget = len(forget_writers)
    fw2local = {w: i for i, w in enumerate(forget_writers)}

    d_model = model.ved.config.encoder.hidden_size
    probe = AdversarialProbe(d_model, n_forget).to(DEVICE)

    f_iter = _infinite(DataLoader(WDataset(forget_df, vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))
    r_iter = _infinite(DataLoader(WDataset(retain_df, vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))

    opt = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()),
                            lr=cfg.UNLEARN_LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP and DEVICE == "cuda")
    checkpoint_steps = set(checkpoint_steps)
    saved = []
    last_probe_acc = float("nan")

    model.train(); probe.train()
    for it in range(1, iters + 1):
        if acfg.PROBE_RESET_EVERY and it % acfg.PROBE_RESET_EVERY == 0:
            probe.reset_parameters()

        bf, br = next(f_iter), next(r_iter)
        opt.zero_grad(set_to_none=True)

        # ---- forget batch: adversarial representation attack + head suppression ----
        pv_f = bf["pixel_values"].to(DEVICE); lb_f = bf["labels"].to(DEVICE)
        wid_f = bf["wid"].to(DEVICE)
        local_f = torch.tensor([fw2local[int(w)] for w in wid_f.detach().cpu().tolist()],
                               dtype=torch.long, device=DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.AMP and DEVICE == "cuda"):
            enc_f = model.ved.encoder(pixel_values=pv_f, return_dict=True)
            z_f = enc_f.last_hidden_state[:, 0, :]
            wlog_f = model.writer_head(z_f)
            rec_f = model.ved(encoder_outputs=enc_f, labels=lb_f, return_dict=True)

            probe_logits = probe(grad_reverse(z_f.float(), acfg.GRL_LAMBDA))
            l_adv = F.cross_entropy(probe_logits, local_f)

            logp_head = F.log_softmax(wlog_f.float(), dim=-1)
            uniform = torch.full_like(logp_head, 1.0 / n_classes)
            l_head = F.kl_div(logp_head, uniform, reduction="batchmean")

            loss1 = (acfg.LAMBDA_ADV * l_adv + acfg.LAMBDA_HEAD * l_head
                     + acfg.LAMBDA_REC * rec_f.loss.float())
        with torch.no_grad():
            last_probe_acc = (probe_logits.detach().argmax(-1) == local_f).float().mean().item() * 100
        scaler.scale(loss1).backward()
        del pv_f, lb_f, enc_f, z_f, wlog_f, rec_f, probe_logits

        # ---- retain batch: identical to unlearn() ----
        pv_r = br["pixel_values"].to(DEVICE); lb_r = br["labels"].to(DEVICE)
        wr_r = br["wid"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.float16,
                                enabled=cfg.AMP and DEVICE == "cuda"):
            rec_r, wlog_r = model(pv_r, lb_r)
            l_retain = F.cross_entropy(wlog_r.float(), wr_r)
            loss2 = acfg.LAMBDA_RETAIN * l_retain + acfg.LAMBDA_REC * rec_r.loss.float()
        scaler.scale(loss2).backward()
        del pv_r, lb_r, rec_r

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), 1.0)
        scaler.step(opt); scaler.update()

        if masks is not None and PConfig.FREEZE_PRUNED:
            apply_masks(model, masks)

        if it % cfg.EVAL_EVERY == 0 or it == iters or it in checkpoint_steps:
            wacc_f = writer_acc(model, forget_df, vocab, fe, n=cfg.EVAL_N)
            wacc_r = writer_acc(model, retain_df, vocab, fe, n=cfg.EVAL_N)
            rq_f = rec_quality(model, forget_df, vocab, fe, n=cfg.CER_EVAL_N)
            print(f"  [{tag}] it {it:>4}  wacc_forget {wacc_f:6.2f}%  "
                  f"wacc_retain {wacc_r:6.2f}%  CER_forget {rq_f['CER']:6.2f}%  "
                  f"train-time_probe_acc {last_probe_acc:6.2f}%")
            model.train(); probe.train()

        if it in checkpoint_steps:
            path = f"{PConfig.OUT}/unlearned_{tag}_step{it}.pt"
            torch.save({"model": model.state_dict(), "iter": it, "method": "arf", "tag": tag},
                       path)
            print(f"    saved -> {path}")
            saved.append(path)

    return model, saved


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B -- MGDA-STYLE DYNAMIC GRADIENT WEIGHTING
#
# RFU-SS's second contribution (parameter self-sharing, Eq. 11-14) is
# architecture-agnostic: a closed-form Pareto-stationary blend weight between
# a forgetting-objective gradient and a remembering-objective gradient on the
# SAME shared parameters, replacing hand-picked fixed weights. Applied here to
# WIC and Random Labeling ONLY -- those two already combine a per-step forget
# loss and a per-step retain loss exactly like RFU-SS's Eq. 11 structure.
# Fisher/Amnesiac/DELETE are two-phase shock-then-repair methods with no
# per-step joint objective to reweight, so they are deliberately left out
# rather than forced into a structure they don't have. Kept as a SEPARATE
# experiment from ARF: this tests only whether dynamic weighting improves the
# existing OUTPUT-level methods, not whether attacking the representation
# helps -- the two questions should not be conflated in one result.
# ══════════════════════════════════════════════════════════════════════════════
def _zero_grad(params):
    for p in params:
        p.grad = None


def compute_grad_vec(loss, params):
    _zero_grad(params)
    loss.backward()
    g = torch.cat([
        (p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1))
        for p in params
    ])
    _zero_grad(params)
    return g


def mgda_alpha(g_forget, g_retain):
    """Closed-form solution to RFU-SS Eq. (14): the Pareto-stationary blend
    weight alpha_u for combining a forgetting-objective gradient (L_u) and a
    remembering-objective gradient (L_r) on shared parameters."""
    diff = g_retain - g_forget
    denom = diff.dot(diff)
    if denom.item() < 1e-12:
        return 0.5
    alpha = diff.dot(g_retain) / denom
    return float(alpha.clamp(0.0, 1.0).item())


def unlearn_mgda(model, ctx, masks, method, iters, checkpoint_steps, tag, cfg=PConfig):
    """WIC / Random Labeling, combined every step via MGDA weighting instead
    of fixed LAMBDA_FORGET/LAMBDA_RETAIN. Runs in fp32 (no AMP): the alpha
    weighting is a stability-sensitive dot-product ratio over ~334M-dim
    gradient vectors, and mixing that with fp16-scaled grads is not worth the
    risk for a batch-size-8, 800-iteration run."""
    assert method in ("wic", "random_label")
    vocab, fe = ctx["vocab"], ctx["fe"]
    forget_df, retain_df = ctx["forget_df"], ctx["retain_df"]
    n_classes = ctx["n_classes"]

    forget_idx_set = set(int(i) for i in forget_df["wid"].unique())
    retain_idx_pool = np.array([i for i in range(n_classes) if i not in forget_idx_set])

    f_iter = _infinite(DataLoader(WDataset(forget_df, vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))
    r_iter = _infinite(DataLoader(WDataset(retain_df, vocab, fe), batch_size=cfg.UNLEARN_BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=S.NUM_WORKERS))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.UNLEARN_LR, weight_decay=0.01)
    params = [p for p in model.parameters() if p.requires_grad]
    checkpoint_steps = set(checkpoint_steps)
    saved = []
    alpha_log = []

    model.train()
    for it in range(1, iters + 1):
        bf, br = next(f_iter), next(r_iter)

        pv_f = bf["pixel_values"].to(DEVICE); lb_f = bf["labels"].to(DEVICE)
        rec_f, wlog_f = model(pv_f, lb_f)
        if method == "wic":
            logp_f = F.log_softmax(wlog_f.float(), dim=-1)
            uniform = torch.full_like(logp_f, 1.0 / n_classes)
            l_forget_head = F.kl_div(logp_f, uniform, reduction="batchmean")
        else:
            rnd_np = np.random.choice(retain_idx_pool, size=wlog_f.size(0))
            rnd = torch.tensor(rnd_np, dtype=torch.long, device=DEVICE)
            l_forget_head = F.cross_entropy(wlog_f.float(), rnd)
        L_u = cfg.LAMBDA_FORGET * l_forget_head + cfg.LAMBDA_REC * rec_f.loss.float()
        g_u = compute_grad_vec(L_u, params)
        del pv_f, lb_f, rec_f, wlog_f

        pv_r = br["pixel_values"].to(DEVICE); lb_r = br["labels"].to(DEVICE)
        wr_r = br["wid"].to(DEVICE)
        rec_r, wlog_r = model(pv_r, lb_r)
        l_retain = F.cross_entropy(wlog_r.float(), wr_r)
        L_r = cfg.LAMBDA_RETAIN * l_retain + cfg.LAMBDA_REC * rec_r.loss.float()
        g_r = compute_grad_vec(L_r, params)
        del pv_r, lb_r, rec_r

        alpha = mgda_alpha(g_u, g_r)
        alpha_log.append(alpha)
        g_final = alpha * g_u + (1 - alpha) * g_r

        opt.zero_grad(set_to_none=True)
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = g_final[offset:offset + n].view_as(p).clone()
            offset += n
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if masks is not None and PConfig.FREEZE_PRUNED:
            apply_masks(model, masks)

        if it % cfg.EVAL_EVERY == 0 or it == iters or it in checkpoint_steps:
            wacc_f = writer_acc(model, forget_df, vocab, fe, n=cfg.EVAL_N)
            wacc_r = writer_acc(model, retain_df, vocab, fe, n=cfg.EVAL_N)
            rq_f = rec_quality(model, forget_df, vocab, fe, n=cfg.CER_EVAL_N)
            print(f"  [{tag}] it {it:>4}  wacc_forget {wacc_f:6.2f}%  "
                  f"wacc_retain {wacc_r:6.2f}%  CER_forget {rq_f['CER']:6.2f}%  "
                  f"alpha_u {alpha:5.3f}")
            model.train()

        if it in checkpoint_steps:
            path = f"{PConfig.OUT}/unlearned_{tag}_step{it}.pt"
            torch.save({"model": model.state_dict(), "iter": it, "method": f"{method}_mgda",
                       "tag": tag, "alpha_log": alpha_log}, path)
            print(f"    saved -> {path}")
            saved.append(path)

    return model, saved


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION -- ARF + MGDA-WIC + MGDA-RandomLabel, starting from a FRESH
# Stage-I prune (deterministic given the same seed/data, so this reproduces
# the same masks as the original 5-method run without needing to chain to
# that kernel's output).
# ══════════════════════════════════════════════════════════════════════════════
ARF_MGDA_SMOKE_TEST = False   # smoke test (10 it, all 3 methods) passed cleanly -- real run


def main_arf_mgda():
    ctx = setup()
    vocab, fe, n_classes = ctx["vocab"], ctx["fe"], ctx["n_classes"]
    results = []
    all_checkpoints = []

    model = build_model(vocab, n_classes)
    load_into(model, ctx["base_sd"], "baseline", strict_report=True)
    results.append(full_report(model, ctx, "BASELINE (multi-head, nothing unlearned)"))

    masks = prune_by_ratio(model, ctx["forget_df"], ctx["retain_df"], vocab, fe)
    results.append(full_report(model, ctx, "AFTER STAGE I (pruning only)"))
    pruned_state = copy.deepcopy(model.state_dict())
    del model; torch.cuda.empty_cache()

    if ARF_MGDA_SMOKE_TEST:
        iters, ckpts = 10, (10,)
    else:
        iters, ckpts = 800, (800,)

    # ---- ARF ----
    m = _fresh_pruned_model(vocab, n_classes, pruned_state)
    m, saved = adversarial_representation_forget(m, ctx, masks, iters=iters,
                                                  checkpoint_steps=ckpts, tag="arf")
    results.append(full_report(m, ctx, f"Prune + ARF ({iters} it)"))
    all_checkpoints += saved
    del m; torch.cuda.empty_cache()

    # ---- WIC + MGDA ----
    m = _fresh_pruned_model(vocab, n_classes, pruned_state)
    m, saved = unlearn_mgda(m, ctx, masks, method="wic", iters=iters,
                            checkpoint_steps=ckpts, tag="wic_mgda")
    results.append(full_report(m, ctx, f"Prune + WIC-MGDA ({iters} it)"))
    all_checkpoints += saved
    del m; torch.cuda.empty_cache()

    # ---- Random Labeling + MGDA ----
    m = _fresh_pruned_model(vocab, n_classes, pruned_state)
    m, saved = unlearn_mgda(m, ctx, masks, method="random_label", iters=iters,
                            checkpoint_steps=ckpts, tag="rl_mgda")
    results.append(full_report(m, ctx, f"Prune + RL-MGDA ({iters} it)"))
    all_checkpoints += saved
    del m; torch.cuda.empty_cache()

    out = pd.DataFrame(results)
    out.to_csv(f"{PConfig.OUT}/comparison_table_arf_mgda.csv", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nCheckpoints saved: {all_checkpoints}")
    print(f"Comparison table -> {PConfig.OUT}/comparison_table_arf_mgda.csv")
    return results

