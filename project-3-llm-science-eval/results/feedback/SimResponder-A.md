# Feedback for SimResponder-A

96 responses graded: 80 correct, 7 partial, 9 incorrect (mean score 0.88).

## Prioritised issues

### 1. `contradicts_accepted_science` (1 responses, 1% of all)
Most affected questions: C01

Highest priority. The answer asserts something that contradicts accepted practice. Add counter-example training items for these topics, require the response to state caveats, and review whether the model is repeating a common misconception.

### 2. `missing_key_point` (5 responses, 5% of all)
Most affected questions: C04, C06

Mostly right but one required point is missing. Use the rubric group coverage to see which point is dropped and add examples where that point is essential.

### 3. `method_or_arithmetic_error` (4 responses, 4% of all)
Most affected questions: N02, N09, N13, N18

Wrong formula or arithmetic. Compare against the listed common mistaken approaches for the affected questions and add worked-solution examples that show the correct formula and why the tempting alternative fails.

### 4. `power_of_ten_error` (4 responses, 4% of all)
Most affected questions: N02, N03, N08

Factor-of-10^k errors point to unit-prefix slips. Add paired items that differ only by prefix and prompt a plausibility check (is this potency realistic for the assay range?).

### 5. `approximation_error` (2 responses, 2% of all)
Most affected questions: N03, N17

Close but outside tolerance, usually premature rounding. Encourage carrying full precision through intermediate steps and rounding only the final answer.

## Weakest topic areas

- Potency and assay math: mean score 0.67
- Assay preparation: mean score 0.75