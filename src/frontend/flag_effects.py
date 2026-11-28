"""Count-model check of the Promotion and Holiday flags: Poisson, negative binomial and zero-inflated models.

Each outcome is modelled from the two flags only (no other columns), so the ratios are the raw effect of each flag.
A zero-inflated model only makes sense when the outcome has many more zeros than a Poisson allows, so it is fitted on
Units_Sold (no zeros in this data) and on Returns (about half zeros). Needs the statsmodels package.
"""
import numpy as np
import pandas as pd

FLAGS = ["Promotion", "Holiday"]


def _yes(s):
    return s.astype(str).str.strip().str.lower().isin(["yes", "y", "1", "true"]).astype(float)


def _fit_all(y, X):
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson
    fits = {}
    pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    fits["Poisson"] = (pois, float(pois.pearson_chi2 / pois.df_resid))
    fits["Negative binomial"] = (sm.NegativeBinomial(y, X).fit(disp=0, maxiter=300), None)
    for name, model in (("Zero-inflated Poisson", ZeroInflatedPoisson(y, X, exog_infl=X, inflation="logit")),
                        ("Zero-inflated neg. binomial", ZeroInflatedNegativeBinomialP(y, X, exog_infl=X, inflation="logit"))):
        try:
            res = model.fit(method="bfgs", maxiter=1000, disp=0)
            if not np.isfinite(res.llf):
                raise ValueError("did not converge")
            fits[name] = (res, None)
        except Exception as e:
            fits[name] = (None, str(e)[:60])
    return fits


def analyze(df, outcomes=("Units_Sold", "Returns")):
    """Returns (comparison table, effects table). df needs Promotion, Holiday (Yes/No) and the outcome columns."""
    X = pd.DataFrame({"const": 1.0, **{f: _yes(df[f]) for f in FLAGS}})
    comp, eff = [], []
    for out in outcomes:
        if out not in df:
            continue
        y = pd.to_numeric(df[out], errors="coerce").fillna(0)
        zeros = float((y == 0).mean())
        for name, (res, extra) in _fit_all(y, X).items():
            if res is None:
                comp.append({"Outcome": out, "Model": name, "Zeros": zeros, "AIC": np.nan, "Note": f"failed to fit ({extra})"})
                continue
            note = f"variance / mean = {extra:.1f}" if name == "Poisson" else ""
            if name.startswith("Zero-inflated"):
                p0 = float(np.mean(res.predict(which="prob-zero"))) if hasattr(res, "predict") else np.nan
                note = f"extra-zero probability {max(p0 - float(np.mean(np.exp(-res.predict(which='mean-main')))), 0):.1%}" if name == "Zero-inflated Poisson" else ""
            comp.append({"Outcome": out, "Model": name, "Zeros": zeros, "AIC": float(res.aic), "Note": note})
            ci = res.conf_int()
            for flag in FLAGS:
                for prefix, part in (("", "Count: rate ratio"), ("inflate_", "Zero part: odds ratio")):
                    key = prefix + flag
                    if key in res.params.index and (prefix == "" or (name.startswith("Zero-inflated") and zeros > 0)):
                        eff.append({"Outcome": out, "Model": name, "Part": part, "Flag": flag, "Ratio": float(np.exp(np.clip(res.params[key], -30, 30))),
                                    "Low": float(np.exp(np.clip(ci.loc[key, 0], -30, 30))), "High": float(np.exp(np.clip(ci.loc[key, 1], -30, 30))), "p-value": float(res.pvalues[key])})
    comp, eff = pd.DataFrame(comp), pd.DataFrame(eff)
    comp["Best"] = comp["AIC"] == comp.groupby("Outcome")["AIC"].transform("min")
    return comp, eff
