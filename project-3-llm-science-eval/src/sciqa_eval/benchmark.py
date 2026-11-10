"""The benchmark: 18 calculation questions and 6 concept questions.

* **Numeric questions** carry an answer computed by :mod:`sciqa_eval.reference`,
  a tolerance, the expected unit, and a list of *plausible wrong answers* that
  correspond to real, common mistakes (wrong formula, forgotten conversion...).
* **Concept questions** carry a rubric: groups of acceptable keyword
  alternatives that must all be covered, plus patterns that signal a
  scientifically wrong claim. Each group also has example sentences used by the
  response simulator (``sentences`` match the rubric; ``paraphrase`` expresses the
  same idea with none of the rubric keywords, to measure how often a keyword
  rubric misses valid wording).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import reference as ref


@dataclass
class Question:
    qid: str
    category: str
    kind: str  # "numeric" | "concept"
    prompt: str
    # numeric
    answer: float | None = None
    unit: str | None = None  # canonical unit the answer is expected in (None = dimensionless)
    display_unit: str = ""  # how a response would typically write the unit
    rel_tol: float = 0.02
    abs_tol: float = 0.0
    partial_rel: float = 0.10  # relative error band that earns partial credit (0 = none)
    given: str = ""
    method: str = ""
    wrong_values: list = field(default_factory=list)  # plausible mistaken answers
    # concept
    groups: list = field(default_factory=list)  # [{"alts": [...], "sentences": [...], "paraphrase": str}]
    forbidden: list = field(default_factory=list)  # regex patterns for wrong claims
    wrong_text: str = ""
    negated_text: str = ""


def _numeric_questions() -> list[Question]:
    N = []

    def add(**kw):
        N.append(Question(kind="numeric", **kw))

    add(qid="N01", category="Potency and assay math",
        prompt="A compound has an IC50 of 25 nM. What is its pIC50?",
        answer=ref.pic50_from_ic50_nm(25), abs_tol=0.02, partial_rel=0.03,
        given="IC50 = 25 nM", method="pIC50 = 9 - log10(IC50 in nM)", wrong_values=[1.40])
    add(qid="N02", category="Potency and assay math",
        prompt="A compound has a pIC50 of 6.3. What is its IC50 in nM?",
        answer=ref.ic50_nm_from_pic50(6.3), unit="nM", display_unit="nM",
        given="pIC50 = 6.3", method="IC50 (nM) = 10^(9 - pIC50)", wrong_values=[0.501])
    add(qid="N03", category="Enzyme kinetics and binding",
        prompt=("An enzyme assay gives an IC50 of 120 nM for a competitive inhibitor, measured at "
                "[S] = 50 uM with Km = 10 uM. What is the Ki in nM?"),
        answer=ref.cheng_prusoff_ki(120, 50, 10), unit="nM", display_unit="nM",
        given="IC50 = 120 nM, [S] = 50 uM, Km = 10 uM", method="Ki = IC50 / (1 + [S]/Km)",
        wrong_values=[100.0])
    add(qid="N04", category="Assay quality and statistics",
        prompt=("In a screening assay the positive control has mean 100 and SD 5, and the negative control "
                "has mean 10 and SD 4. What is the Z'-factor?"),
        answer=ref.z_prime(100, 5, 10, 4), abs_tol=0.01, partial_rel=0.15,
        given="positive 100 +/- 5, negative 10 +/- 4", method="Z' = 1 - 3(SDp + SDn)/|mean_p - mean_n|",
        wrong_values=[0.9])
    add(qid="N05", category="Drug-likeness and efficiency",
        prompt=("A ligand has pIC50 = 7.0 and 28 heavy atoms. Using LE = 1.37 x pIC50 / heavy atoms, "
                "what is its ligand efficiency in kcal/mol per heavy atom?"),
        answer=ref.ligand_efficiency(7.0, 28), unit="kcal/mol", display_unit="kcal/mol per heavy atom",
        given="pIC50 = 7.0, 28 heavy atoms", method="LE = 1.37 * pIC50 / HA", wrong_values=[0.25])
    add(qid="N06", category="Drug-likeness and efficiency",
        prompt="A compound has pIC50 = 7.5 and logP = 3.1. What is its lipophilic ligand efficiency (LLE)?",
        answer=ref.lipophilic_ligand_efficiency(7.5, 3.1), abs_tol=0.05,
        given="pIC50 = 7.5, logP = 3.1", method="LLE = pIC50 - logP", wrong_values=[10.6])
    add(qid="N07", category="Omics statistics",
        prompt=("A gene has mean normalised counts of 80 in treated samples and 20 in controls. "
                "What is the log2 fold change (treated vs control)?"),
        answer=ref.log2_fold_change(80, 20), abs_tol=0.05, partial_rel=0.2,
        given="treated = 80, control = 20", method="log2FC = log2(treated / control)", wrong_values=[4.0])
    add(qid="N08", category="Omics statistics",
        prompt=("Six genes have p-values 0.001, 0.008, 0.039, 0.041, 0.042 and 0.060. What is the "
                "Benjamini-Hochberg adjusted p-value of the gene with p = 0.039?"),
        answer=ref.benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042, 0.060])[2],
        given="p = 0.001, 0.008, 0.039, 0.041, 0.042, 0.060 (m = 6)",
        method="adjusted p = min over larger ranks of p * m / rank", wrong_values=[0.234])
    add(qid="N09", category="Omics statistics",
        prompt=("A differential-expression study tests 20,000 genes at a family-wise alpha of 0.05. "
                "What is the Bonferroni per-test significance threshold?"),
        answer=ref.bonferroni_threshold(0.05, 20000),
        given="alpha = 0.05, m = 20000", method="threshold = alpha / m", wrong_values=[1000.0])
    add(qid="N10", category="Omics statistics",
        prompt=("A background of 20,000 genes contains 500 pathway genes. A hit list of 300 genes contains 25 "
                "pathway genes. What is the hypergeometric enrichment p-value, P(X >= 25)?"),
        answer=ref.hypergeometric_enrichment_p(20000, 500, 300, 25), rel_tol=0.05, partial_rel=0.25,
        given="N = 20000, K = 500, n = 300, k = 25", method="P(X >= k) from the hypergeometric distribution",
        wrong_values=[7.5])
    add(qid="N11", category="Enzyme kinetics and binding",
        prompt=("A ligand binds with Kd = 10 nM at 298.15 K. Using dG = RT ln(Kd) with "
                "R = 0.001987 kcal/mol/K, what is the binding free energy in kcal/mol?"),
        answer=ref.delta_g_binding_kcal(10e-9), unit="kcal/mol", display_unit="kcal/mol",
        abs_tol=0.15, partial_rel=0.05,
        given="Kd = 10 nM = 1e-8 M, T = 298.15 K", method="dG = RT ln(Kd)", wrong_values=[-4.74])
    add(qid="N12", category="Enzyme kinetics and binding",
        prompt="At a ligand concentration of 50 nM and Kd = 10 nM, what fraction of receptor is occupied?",
        answer=ref.fractional_occupancy(50, 10), abs_tol=0.01,
        given="[L] = 50 nM, Kd = 10 nM", method="occupancy = [L] / ([L] + Kd)", wrong_values=[0.167])
    add(qid="N13", category="Assay preparation",
        prompt="How many uL of a 10 mM stock are needed to prepare 100 uL of a 50 uM solution?",
        answer=ref.stock_volume(10_000, 50, 100), unit="uL", display_unit="uL",
        given="stock = 10 mM = 10000 uM, final = 50 uM, volume = 100 uL", method="C1V1 = C2V2",
        wrong_values=[1.0])
    add(qid="N14", category="Chemistry basics",
        prompt=("What is the molecular weight of C15H12ClN3O2? "
                "(C 12.011, H 1.008, Cl 35.45, N 14.007, O 15.999)"),
        answer=ref.molecular_weight({"C": 15, "H": 12, "Cl": 1, "N": 3, "O": 2}), unit="g/mol",
        display_unit="g/mol", rel_tol=0.01, partial_rel=0.0,  # exact arithmetic: no partial credit
        given="C15 H12 Cl N3 O2", method="sum of atomic masses",
        wrong_values=[289.6])
    add(qid="N15", category="Drug-likeness and efficiency",
        prompt=("A compound has MW 540, logP 5.8, 2 H-bond donors and 9 H-bond acceptors. "
                "How many Lipinski rule-of-five criteria does it violate?"),
        answer=float(ref.lipinski_violations(540, 5.8, 2, 9)), abs_tol=0.01, partial_rel=0.0,
        given="MW 540, logP 5.8, HBD 2, HBA 9", method="count MW>500, logP>5, HBD>5, HBA>10",
        wrong_values=[1.0])
    add(qid="N16", category="Pharmacology",
        prompt=("An inhibitor with Hill slope 1 is applied at 3 x its IC50. "
                "What fraction of activity is inhibited?"),
        answer=ref.hill_fractional_inhibition(3.0), abs_tol=0.01, partial_rel=0.1,
        given="[I]/IC50 = 3, Hill = 1", method="inhibition = x / (1 + x)", wrong_values=[0.5])
    add(qid="N17", category="Pharmacology",
        prompt=("A compound has IC50 = 15 nM on its target and 4500 nM on an off-target. "
                "What is the selectivity (fold)?"),
        answer=ref.selectivity_fold(4500, 15), display_unit="fold",
        given="on-target 15 nM, off-target 4500 nM", method="selectivity = IC50(off) / IC50(on)",
        wrong_values=[0.00333])
    add(qid="N18", category="Potency and assay math",
        prompt=("Three replicate IC50 measurements are 10 nM, 100 nM and 1000 nM. What is the appropriate "
                "average potency expressed as pIC50 (averaging in log space)?"),
        answer=ref.mean_pic50_from_ic50_nm([10, 100, 1000]), abs_tol=0.02, partial_rel=0.05,
        given="IC50 = 10, 100, 1000 nM", method="mean of the pIC50 values", wrong_values=[6.43])
    return N


def _concept_questions() -> list[Question]:
    def g(alts, sentences, paraphrase):
        return {"alts": alts, "sentences": sentences, "paraphrase": paraphrase}

    Q = []
    Q.append(Question(
        qid="C01", category="Concepts: enzyme kinetics", kind="concept",
        prompt="How does the IC50 of a competitive inhibitor relate to its Ki?",
        groups=[
            g(["cheng-prusoff", "cheng prusoff"],
              ["The two are linked by the Cheng-Prusoff equation.",
               "The conversion is given by the Cheng-Prusoff relationship."],
              "There is a standard textbook equation that converts one into the other."),
            g(["substrate concentration", "[s]"],
              ["The measured IC50 rises as the substrate concentration increases, because substrate competes with the inhibitor.",
               "IC50 shifts whenever the substrate concentration in the assay is changed."],
              "A higher amount of competing substrate makes the inhibitor look weaker."),
            g(["km"],
              ["The size of the correction is set by the substrate level relative to Km.",
               "Km, the Michaelis constant, determines how strongly the substrate competes."],
              "The size of the correction depends on the enzyme's affinity for its substrate."),
            g(["dissociation constant", "intrinsic", "independent of substrate"],
              ["Ki is the inhibitor's dissociation constant, a property of the enzyme-inhibitor pair.",
               "Ki is intrinsic to the inhibitor and enzyme, unlike IC50."],
              "Ki reflects true binding affinity and does not change with assay conditions."),
        ],
        forbidden=[r"ic50\s+(is\s+)?(always\s+)?(equal to|equals|the same as)\s+(the\s+)?ki"],
        wrong_text="For a competitive inhibitor, IC50 is always equal to Ki, regardless of how much substrate is present.",
        negated_text="It is a common misconception that IC50 equals Ki; they coincide only when substrate is far below Km.",
    ))
    Q.append(Question(
        qid="C02", category="Concepts: data and modelling", kind="concept",
        prompt="Why can random train/test splits overestimate QSAR model performance, and what is a better approach?",
        groups=[
            g(["analogs", "analogues", "same chemical series", "near-duplicates", "similar compounds"],
              ["Random splits put close analogs of every test compound into the training set.",
               "Compounds from the same chemical series end up on both sides of the split."],
              "Test molecules have near twins in the training data, so the test is too easy."),
            g(["optimistic", "overestimate", "inflate"],
              ["The resulting performance estimate is optimistic.",
               "Reported accuracy is inflated compared with real prospective use."],
              "The numbers look better than they will be in practice."),
            g(["scaffold split", "cluster split", "time split", "temporal split", "leave-one-series-out", "group split"],
              ["A scaffold split keeps whole chemical series out of the training data.",
               "A time split, training on older compounds and testing on newer ones, mimics prospective use."],
              "Holding out entire families of molecules gives a more honest estimate."),
            g(["new chemotype", "novel scaffold", "extrapolat", "applicability domain"],
              ["What matters in a project is accuracy on new chemotypes, which requires extrapolation.",
               "Predictions outside the applicability domain should be treated with caution."],
              "Real projects need predictions for chemistry the model has never seen."),
        ],
        forbidden=[r"gold standard", r"always gives?\s+an unbiased"],
        wrong_text="Random splitting is the gold standard and always gives an unbiased estimate of real-world performance.",
        negated_text="It is false that random splitting is the gold standard for prospective performance.",
    ))
    Q.append(Question(
        qid="C03", category="Concepts: drug-likeness", kind="concept",
        prompt="Does passing Lipinski's rule of five guarantee that a compound is orally bioavailable?",
        groups=[
            g(["not a guarantee", "does not guarantee", "no guarantee", "rule of thumb", "guideline"],
              ["Passing the rule is not a guarantee of oral bioavailability.",
               "The rule of five is a rule of thumb, not a law."],
              "Meeting the criteria does not ensure that a compound is absorbed well."),
            g(["molecular weight", "hydrogen-bond", "hydrogen bond", "logp"],
              ["It flags molecular weight above 500, logP above 5, more than 5 donors or more than 10 acceptors.",
               "The criteria concern hydrogen bond donors and acceptors, size and lipophilicity."],
              "It is based on simple counts of size, lipophilicity, donors and acceptors."),
            g(["permeability", "solubility", "first-pass", "transporter", "efflux"],
              ["Bioavailability also depends on solubility and membrane permeability.",
               "First-pass metabolism, efflux and transporters further reduce exposure."],
              "Absorption and liver clearance matter as well."),
            g(["exception", "natural product", "macrocycle", "beyond rule of five", "bro5"],
              ["There are many exceptions among approved drugs.",
               "Natural products and macrocycles often violate the rule yet are orally active."],
              "Plenty of marketed oral drugs break these limits."),
        ],
        forbidden=[r"guarantee[sd]?\s+(to be\s+|that it is\s+)?orally", r"is guaranteed to be bioavailable"],
        wrong_text="Any compound that passes Lipinski's rule of five is guaranteed to be orally bioavailable.",
        negated_text="It is a myth that a compound passing the rule is guaranteed to be orally bioavailable.",
    ))
    Q.append(Question(
        qid="C04", category="Concepts: data quality", kind="concept",
        prompt="Why are mixed concentration units in a bioactivity table a data-quality risk, and what should be done?",
        groups=[
            g(["nm, um", "nm and um", "nm or um", "nanomolar", "micromolar"],
              ["Records report potencies in nM, uM or even M, so the same number can mean very different things.",
               "Some values are nanomolar and others micromolar."],
              "Different rows use different concentration scales for the same assay."),
            g(["1000", "orders of magnitude", "three log"],
              ["A mislabelled unit shifts the apparent potency 1000-fold.",
               "The error can reach three orders of magnitude."],
              "The mistake can be enormous in potency terms."),
            g(["standardize", "standardise", "convert", "normalize", "normalise"],
              ["Convert every record to one unit, such as nM, before comparing values.",
               "Standardise to a single unit and compute pIC50."],
              "Put everything on one common scale first."),
            g(["flag", "range check", "outlier", "replicate", "sanity check"],
              ["Add automated checks that flag implausible values and replicate disagreement.",
               "Range checks against the assay's dynamic range catch impossible values."],
              "Automated screening for impossible numbers is worthwhile."),
        ],
        forbidden=[r"mixed units are harmless", r"units (do not|don't) matter"],
        wrong_text="Mixed units are harmless because the numbers can simply be averaged directly.",
        negated_text="It is not true that mixed units are harmless.",
    ))
    Q.append(Question(
        qid="C05", category="Concepts: screening", kind="concept",
        prompt="What are PAINS compounds and how should they be treated in a screening hit list?",
        groups=[
            g(["pan-assay", "pan assay"],
              ["PAINS stands for pan-assay interference compounds.",
               "PAINS are 'pan assay' interference chemotypes."],
              "PAINS is an acronym for a class of troublesome screening compounds."),
            g(["false positive", "promiscuous", "assay artifact", "artefact", "artifact"],
              ["They tend to give false positives by reacting with or disrupting many different assays.",
               "Their activity is often an assay artifact rather than genuine target binding."],
              "Their apparent activity usually does not reflect real target engagement."),
            g(["substructure", "structural alert", "structural filter"],
              ["They are recognised by substructure filters such as catechols or rhodanines.",
               "Structural alerts are applied computationally to flag them."],
              "Certain recurring chemical motifs identify them."),
            g(["orthogonal", "counter-screen", "counterscreen", "confirm", "follow-up", "secondary assay"],
              ["Flagged hits should be confirmed in an orthogonal assay before investing further.",
               "A counter-screen helps decide whether activity is real."],
              "They should be re-tested in a different assay format before anyone relies on them."),
        ],
        forbidden=[r"pains\W+(compounds\W+)?(are|is)\W+(guaranteed\W+|always\W+)(true|genuine|real)"],
        wrong_text="PAINS compounds are always genuine hits and should be advanced without further testing.",
        negated_text="It is wrong to assume that PAINS compounds are always genuine hits.",
    ))
    Q.append(Question(
        qid="C06", category="Concepts: omics statistics", kind="concept",
        prompt="Why is a raw p-value threshold of 0.05 not sufficient when testing thousands of genes for differential expression?",
        groups=[
            g(["multiple testing", "multiple comparisons", "thousands of tests", "thousands of genes"],
              ["Testing thousands of genes at once is a multiple testing problem.",
               "With thousands of tests, small p-values arise frequently by chance alone."],
              "Looking at so many genes simultaneously changes how p-values should be read."),
            g(["false positive", "type i", "expected number"],
              ["At alpha 0.05, about 5% of truly null genes would still be false positives.",
               "The expected number of false hits is large without correction."],
              "Many apparently significant genes would simply be noise."),
            g(["false discovery rate", "fdr", "benjamini", "bonferroni", "adjusted p", "q-value"],
              ["Control the false discovery rate with the Benjamini-Hochberg procedure.",
               "Use Bonferroni or report adjusted p-values (q-values)."],
              "Apply a standard correction procedure to the whole list of p-values."),
        ],
        forbidden=[r"no correction is (needed|necessary)", r"p-value below 0\.05 is sufficient"],
        wrong_text="No correction is needed; a raw p-value below 0.05 is sufficient for every gene.",
        negated_text="It is wrong to say that no correction is needed when testing thousands of genes.",
    ))
    return Q


def build_benchmark() -> list[Question]:
    return _numeric_questions() + _concept_questions()


def benchmark_by_id() -> dict[str, Question]:
    return {q.qid: q for q in build_benchmark()}
