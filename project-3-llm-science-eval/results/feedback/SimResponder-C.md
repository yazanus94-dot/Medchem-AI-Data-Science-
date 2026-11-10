# Feedback for SimResponder-C

96 responses graded: 30 correct, 17 partial, 49 incorrect (mean score 0.44).

## Prioritised issues

### 1. `contradicts_accepted_science` (7 responses, 7% of all)
Most affected questions: C04, C01, C02, C03, C05, C06

Highest priority. The answer asserts something that contradicts accepted practice. Add counter-example training items for these topics, require the response to state caveats, and review whether the model is repeating a common misconception.

### 2. `method_or_arithmetic_error` (23 responses, 24% of all)
Most affected questions: N01, N04, N14, N08, N09, N12

Wrong formula or arithmetic. Compare against the listed common mistaken approaches for the affected questions and add worked-solution examples that show the correct formula and why the tempting alternative fails.

### 3. `approximation_error` (9 responses, 9% of all)
Most affected questions: N11, N03, N05, N09, N12, N16

Close but outside tolerance, usually premature rounding. Encourage carrying full precision through intermediate steps and rounding only the final answer.

### 4. `missing_most_key_points` (6 responses, 6% of all)
Most affected questions: C01, C05, C06

The explanation is too thin to be useful. Reward complete, structured answers that address every rubric point, not only a single correct sentence.

### 5. `missing_key_point` (6 responses, 6% of all)
Most affected questions: C02, C04, C03, C05

Mostly right but one required point is missing. Use the rubric group coverage to see which point is dropped and add examples where that point is essential.

### 6. `power_of_ten_error` (4 responses, 4% of all)
Most affected questions: N05, N06, N08, N10

Factor-of-10^k errors point to unit-prefix slips. Add paired items that differ only by prefix and prompt a plausibility check (is this potency realistic for the assay range?).

### 7. `sign_error` (4 responses, 4% of all)
Most affected questions: N05, N06, N08, N12

Sign convention errors (for example binding free energy is negative for favourable binding). Add items that test sign interpretation, not just magnitude.

### 8. `abstention` (4 responses, 4% of all)
Most affected questions: N07, N10, N11, C03

The responder declined to answer a question that is answerable from the information given. Check whether abstention is calibrated, or whether the model is being too cautious.

### 9. `unit_missing` (2 responses, 2% of all)
Most affected questions: N02, N03

The value is right but the unit is missing. Require a unit with every reported quantity in the answer format and in the reward signal.

### 10. `unit_error` (1 responses, 1% of all)
Most affected questions: N03

The number carries a unit of the wrong dimension. Add items that force explicit unit tracking (nM vs uM, uL vs mL, kcal/mol vs kJ/mol) and reward answers that state and convert units correctly.

## Weakest topic areas

- Enzyme kinetics and binding: mean score 0.23
- Assay quality and statistics: mean score 0.25